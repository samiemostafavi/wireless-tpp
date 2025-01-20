import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise


class RecurrentE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(RecurrentE2E, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        logger.info(f"RecurrentE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)

        # Noise regularization, only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)

        self.concat_embeddings = model_config.model_specs['embeddings']['concat']

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.src_seq_len = model_config.model_specs['history']['length']
        self.num_mix_components_dtime = model_config.model_specs['mdn']['num_mix_components_dtime']
        
        self.include_mcs = model_config.model_specs['history']['include_mcs']
        self.mcs_emb_dim = model_config.model_specs['embeddings']['mcs_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_mretx = model_config.model_specs['history']['include_mretx']
        self.mretx_emb_dim = model_config.model_specs['embeddings']['mretx_emb_dim'] if self.concat_embeddings else self.d_model

        assert model_config.model_specs['rnn_type'] == 'lstm' or model_config.model_specs['rnn_type'] == 'gru' or model_config.model_specs['rnn_type'] == 'rnn'
        self.rnn_type = model_config.model_specs['rnn_type']
        self.num_layers = model_config.model_specs['num_layers']
        self.bidirectional = bool(model_config.model_specs['bidirectional'])
        logger.info(f"RNN type: {self.rnn_type}, num_layers: {self.num_layers}, bidirectional: {self.bidirectional}")
        logger.info(f"Include MCS: {self.include_mcs}, Include MRetx: {self.include_mretx}")
        
        self.use_norm = model_config.use_ln

        self.tgt_seq_len = model_config.model_specs['tgt_seq_len']
        self.dropout = model_config.dropout_rate
        
        # History embedding configurations
        # mcs embedding
        self.num_mcs_types = 30  # MCS indices: 0 to 28 (29 types), and padding token
        self.mcs_pad_id = 29
        # retransmissions embedding
        self.num_mretx_types = 5  # retransmission indices: 0 to 3 (4 types), and padding token
        self.mretx_pad_id = 4

        # delay embedding layer
        self.dtime_emb_layer = nn.Linear(1, self.d_model)

        # Embedding layers defenitions
        if self.include_mcs:
            # mcs encoding
            self.layer_mcs_emb = nn.Embedding(
                self.num_mcs_types,
                self.mcs_emb_dim,
                padding_idx=self.mcs_pad_id,
                device=self.device
            )
        if self.include_mretx:
            # retransmissions encoding
            self.layer_mretx_emb = nn.Embedding(
                self.num_mretx_types,
                self.mretx_emb_dim,
                padding_idx=self.mretx_pad_id,
                device=self.device
            )

        if self.rnn_type == 'lstm':
            self.layer_rnn = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.layer_rnn = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'rnn':
            self.layer_rnn = nn.RNN(
                input_size=self.d_model,
                hidden_size=self.d_model,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        
        # prediction linear layer
        if self.bidirectional:
            self.dtime_linear = nn.Linear(2*self.d_model, 3 * self.num_mix_components_dtime)
        else:
            self.dtime_linear = nn.Linear(self.d_model, 3 * self.num_mix_components_dtime)
        

    def get_pred_distribution(self, context) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            context (tensor): [batch_size, tgt_seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution.
        """
        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.dtime_linear(context)

        locs = raw_params[..., :self.num_mix_components_dtime]
        log_scales = raw_params[..., self.num_mix_components_dtime: (2 * self.num_mix_components_dtime)]
        log_weights = raw_params[..., (2 * self.num_mix_components_dtime):]
        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        pred_dtime_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_val=self.mean_dtime,
            std_val=self.std_dtime
        )
        return pred_dtime_dist


    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """

        history_seq_obj = HistorySequence(batch, self.src_seq_len, self.tgt_seq_len, self.dtime_transform)
        target_seq_obj = TargetSequence(batch, self.src_seq_len, self.tgt_seq_len, self.dtime_transform)

        # Shapes:
        # history_seq_obj.dtime_seqs: [batch_size, src_seq_len]
        # target_seq_obj.dtime_seqs:  [batch_size, tgt_seq_len]

        # 2) Concatenate the entire sequence of dtimes: [batch_size, src_seq_len + tgt_seq_len]
        #    This is one common approach so we can run a single pass through the RNN.
        dtime_full = torch.cat(
            [history_seq_obj.dtime_seqs, target_seq_obj.dtime_seqs],
            dim=1
        ) # [batch_size, src_seq_len + tgt_seq_len]

        # apply embedding on the delay sequence
        dtime_full = dtime_full.float()
        dtime_embed = dtime_full.unsqueeze(-1)  # [batch_size, src_seq_len + tgt_seq_len, 1]
        dtime_embed = self.dtime_emb_layer(dtime_embed)  # [batch_size, src_seq_len + tgt_seq_len, d_model]

        # rnn_out: [batch_size, src_seq_len + tgt_seq_len, d_model(*2 if bidirectional)]
        rnn_out, _ = self.layer_rnn(dtime_embed)

        # 5) Slice out the RNN output for the target portion
        #    rnn_out[:, :self.src_seq_len-1, :] -> output of the history portion (but one less, as the output of the last step is considered prediction for the first target step)
        #    rnn_out[:, self.src_seq_len-1:-1, :] -> output of the target portion (shifted by one, as the last step's prediction cannot be used for training, but the last step of the history section is actually the first step of the target section)
        target_out = rnn_out[:, self.src_seq_len-1:-1, :]  # shape = [batch_size, tgt_seq_len, d_model(*2 if bidirectional)]

        # 6) Get mixture distribution parameters for the target portion
        pred_dist = self.get_pred_distribution(target_out)
        # pred_dist.log_prob(...) can handle shape = [batch_size, tgt_seq_len]

        # 7) Compare with the ground truth target dtimes
        #    shape of target_seq_obj.dtime_seqs => [batch_size, tgt_seq_len]
        labels = target_seq_obj.dtime_seqs_transformed
        assert labels.shape == pred_dist.mean.shape
        dtime_ll = pred_dist.log_prob(labels)
        dtime_loss = -dtime_ll.sum()
        # shape => [batch_size, tgt_seq_len]

        # 9) Count how many events we predicted. 
        #    If there's no padding, it's simply B * tgt_seq_len. If there's padding, 
        #    you might do something like (target_seq_obj.dtime_seqs != PAD_VALUE).sum().
        num_predictions = target_seq_obj.dtime_seqs.shape[0] * target_seq_obj.dtime_seqs.shape[1]
        # or if you are zero/pad masking, do:
        # num_events = (target_seq_obj.dtime_seqs != PAD_VALUE).sum()

        return dtime_loss, num_predictions, None, None


    def predict_mean_variance(self, batch, forward=False):
        """
        Predict mean & variance for `tgt_seq_len` future steps, given `src_seq_len` historical data,
        by auto-regressively feeding the predicted *mean* from each step back as the next input.

        Returns:
            all_means: [batch_size, tgt_seq_len]
            all_vars:  [batch_size, tgt_seq_len]
        """
        history_seq_obj = HistorySequence(batch, self.src_seq_len, self.tgt_seq_len, self.dtime_transform)
        target_seq_obj = TargetSequence(batch, self.src_seq_len, self.tgt_seq_len, self.dtime_transform)
        labels = target_seq_obj.dtime_seqs_transformed

        # apply embedding on the delay sequence
        dtime_seqs = history_seq_obj.dtime_seqs.float()
        dtime_embed = dtime_seqs.unsqueeze(-1)  # [batch_size, src_seq_len, 1]
        dtime_embed = self.dtime_emb_layer(dtime_embed)  # [batch_size, src_seq_len, d_model]

        # rnn_out: [batch_size, src_seq_len, d_model(*2 if bidirectional)]
        rnn_out, hidden = self.layer_rnn(dtime_embed)
        step_out = rnn_out[:,-1,:]  # => [batch_size, d_model(*2)]
        step_dist = self.get_pred_distribution(step_out)  # NormalMixtureDistribution
        # step_dist.mean, step_dist.variance => shape [batch_size]
        step_mean = step_dist.mean
        step_var = step_dist.variance

        all_means = [ step_mean ]
        all_vars = [ step_var ]

        # [batch_size]
        current_input = step_mean

        # We'll step one time at a time for the future steps.
        for _ in range(self.tgt_seq_len-1):
            # a) Embed the current_input [batch_size] => shape [batch_size, 1, d_model]
            x_next = self.dtime_emb_layer(
                self.dtime_transform.inv(current_input).unsqueeze(1).unsqueeze(2)
            )
            # now x_next: [batch_size, 1, d_model]

            # call RNN [batch_size, 1, d_model]:
            rnn_out_next, hidden = self.layer_rnn(x_next, hidden) # (hidden[0].squeeze(0), hidden[1].squeeze(0))

            # shape of rnn_out_next => [batch_size, 1, d_model(*2)]
            # updated hidden => updated h(, c)

            # c) Get distribution for this step
            #    shape => [batch_size, 1, hidden_size] => we can squeeze out the dim=1
            step_out = rnn_out_next.squeeze(1)  # => [batch_size, d_model(*2)]
            step_dist = self.get_pred_distribution(step_out)  # NormalMixtureDistribution

            # step_dist.mean, step_dist.variance => shape [batch_size]

            step_mean = step_dist.mean # shape [batch_size]
            step_var = step_dist.variance # shape [batch_size]

            # d) Save them for returning
            all_means.append(step_mean)
            all_vars.append(step_var)

            # e) Feed the newly predicted *mean* back
            current_input = step_mean  # shape [batch_size]

        # 5) Stack them into [batch_size, tgt_seq_len]
        pred_dtime = torch.stack(all_means, dim=1)  # [batch_size, tgt_seq_len]
        pred_dtime_var = torch.stack(all_vars, dim=1)    # [batch_size, tgt_seq_len]

        assert labels.shape == pred_dtime.shape
        assert labels.shape == pred_dtime_var.shape

        num_predictions = labels.shape[0] * labels.shape[1]

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), None, num_predictions


    def predict_probabilities(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        slot_seqs = slot_seqs[:, -1 -self.his_len:]
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        type_seqs = type_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            slot_seqs[:, :-1],
            len_seqs_transformed[:, :-1], 
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        # select the last output
        enc_out = enc_out[:, -1:, :]
        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        if not self.mdn_2d:
            # [batch_size, seq_len, 3 * num_mix_components]
            pred_dtime_dist = self.dtime_distribution(enc_out)

            sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
            sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
            num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
            time_since_last_event = torch.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime, device=self.device)
            dtimes_logprob_pred = pred_dtime_dist.log_prob(time_since_last_event)

            pred_len_dist = self.len_distribution(enc_out)
            sample_len_min = prediction_config['probability_generation']['sample_len_min']
            sample_len_max = prediction_config['probability_generation']['sample_len_max']
            num_steps_len = prediction_config['probability_generation']['num_steps_len']
            len_samples = torch.linspace(sample_len_min, sample_len_max, num_steps_len, device=self.device)
            lens_logprob_pred = pred_len_dist.log_prob(len_samples)

            return (dtimes_logprob_pred, lens_logprob_pred), (dtime_seqs_transformed, time_seqs, type_seqs, slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events
        else:
            pred_joint_dist = self.joint_distribution(enc_out)
            
            # Step 1: Create 1D linspace for each dimension
            sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
            sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
            num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
            sample_len_min = prediction_config['probability_generation']['sample_len_min']
            sample_len_max = prediction_config['probability_generation']['sample_len_max']
            num_steps_len = prediction_config['probability_generation']['num_steps_len']
            dtime_linspace = torch.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime, device=self.device)
            len_linspace = torch.linspace(sample_len_min, sample_len_max, num_steps_len, device=self.device)

            # Step 2: Generate a 2D meshgrid
            time_grid, len_grid = torch.meshgrid(dtime_linspace, len_linspace, indexing="ij")

            # Step 3: Stack to get a grid of 2D samples with shape [num_samples, num_samples, 2]
            sample_grid = torch.stack((time_grid, len_grid), dim=-1)  # Shape: [num_samples, num_samples, 2]

            # Step 4: Reshape to get [num_samples * num_samples, 1, 1, 2]
            sample_grid = sample_grid.reshape(-1, 1, 1, 2)

            # Step 5: Expand to match [num_samples * num_samples, batch_size, seq_len, 2]
            batch_size, seq_len = dtime_seqs_transformed[:, :-1].shape
            sample_grid = sample_grid.expand(-1, batch_size, seq_len, -1)  # Shape: [num_samples * num_samples, batch_size, seq_len, 2]

            # Now sample_grid has the shape [num_samples * num_samples, batch_size, seq_len, 2]
            joint_logpdfs_pred = pred_joint_dist.log_prob(sample_grid)

            return (joint_logpdfs_pred, ), (dtime_seqs_transformed, time_seqs, type_seqs, slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events

        
    def generate_samples(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        slot_seqs = slot_seqs[:, -1 -self.his_len:]
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        type_seqs = type_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            slot_seqs[:, :-1],
            len_seqs_transformed[:, :-1], 
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        # select the last output
        enc_out = enc_out[:, -1:, :]
        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        # [batch_size, seq_len, 3 * num_mix_components]
        pred_dtime_dist = self.dtime_distribution(enc_out)
        dtimes_samples = pred_dtime_dist.sample((prediction_config['num_samples_dtime'],))

        pred_len_dist = self.len_distribution(enc_out)
        len_samples = pred_len_dist.sample((prediction_config['num_samples_len'],))

        return (dtimes_samples, len_samples), (dtime_seqs_transformed, time_seqs, type_seqs, slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events
    


class MarginalE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(MarginalE2E, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        logger.info(f"MarginalE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)

        # Noise regularization, only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)

        self.num_mix_components_dtime = model_config.model_specs['mdn']['num_mix_components_dtime']

        # RETX prediction linear layer
        self.dtime_linear = nn.Parameter(torch.empty(3 * self.num_mix_components_dtime, device=self.device))
        nn.init.uniform_(self.dtime_linear, a=0.0, b=1.0)

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """

        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # we only take the last delay value
        dtimes_transformed = dtime_seqs_transformed[:, -1] # shape: [batch_size]
        dtimes = self.dtime_transform.inv(dtimes_transformed) # shape: [batch_size]

        batch_size = dtimes.shape[0]
        # [batch_size, 3 * num_mix_components]
        raw_params = self.dtime_linear.unsqueeze(0) # -> [1, 3 * num_mix_components]
        raw_params = raw_params.repeat(batch_size, 1) # -> [batch_size, 3 * num_mix_components]

        locs = raw_params[..., :self.num_mix_components_dtime]
        log_scales = raw_params[..., self.num_mix_components_dtime: (2 * self.num_mix_components_dtime)]
        log_weights = raw_params[..., (2 * self.num_mix_components_dtime):]
        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        pred_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_val=self.mean_dtime,
            std_val=self.std_dtime
        )

        # 7) Compare with the ground truth target dtimes
        #    shape of target_seq_obj.dtime_seqs => [batch_size]
        assert dtimes_transformed.shape == pred_dist.mean.shape
        dtime_ll = pred_dist.log_prob(dtimes_transformed)
        dtime_loss = -dtime_ll.sum()
        # shape => [batch_size]

        num_predictions = batch_size

        return dtime_loss, num_predictions, None, None


    def predict_mean_variance(self, batch, forward=False):
        """
        Predict mean & variance for `tgt_seq_len` future steps, given `src_seq_len` historical data,
        by auto-regressively feeding the predicted *mean* from each step back as the next input.

        Returns:
            all_means: [batch_size, tgt_seq_len]
            all_vars:  [batch_size, tgt_seq_len]
        """
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # we only take the last delay value
        dtimes_transformed = dtime_seqs_transformed[:, -1] # shape: [batch_size]
        dtimes = self.dtime_transform.inv(dtimes_transformed) # shape: [batch_size]

        batch_size = dtimes.shape[0]
        # [batch_size, 3 * num_mix_components]
        raw_params = self.dtime_linear.unsqueeze(0) # -> [1, 3 * num_mix_components]
        raw_params = raw_params.repeat(batch_size, 1) # -> [batch_size, 3 * num_mix_components]

        locs = raw_params[..., :self.num_mix_components_dtime]
        log_scales = raw_params[..., self.num_mix_components_dtime: (2 * self.num_mix_components_dtime)]
        log_weights = raw_params[..., (2 * self.num_mix_components_dtime):]
        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        pred_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_val=self.mean_dtime,
            std_val=self.std_dtime
        )
        pred_dtime = pred_dist.mean.unsqueeze(1)  # [batch_size, 1]
        pred_dtime_var = pred_dist.variance.unsqueeze(1)  # [batch_size, 1]
        num_predictions = batch_size
        labels = dtimes_transformed.unsqueeze(1)  # [batch_size, 1]

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), None, num_predictions


class HistorySequence():
    # the data sequence is divided into two parts: history and target
    # [:, -self.tgt_seq_len:] is the target sequence with the length of self.tgt_seq_len
    # target sequence consists of only departure events type = max_type
    # [:, -2-self.src_seq_len:-self.tgt_seq_len] is the history sequence with the length: self.src_seq_len
    # history sequence does not include departure events
    def __init__(self, batch, src_seq_len, tgt_seq_len, dtime_transform):
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch
        self.dtime_transform = dtime_transform

        # only consider the last self.src_seq_len events in the history
        self.slot_seqs = slot_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.len_seqs_transformed = len_seqs_transformed[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.mcs_seqs = mcs_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.mretx_seqs = mretx_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.rfailed_seqs = rfailed_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.num_rbs_seqs = num_rbs_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.time_seqs = time_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.type_seqs = type_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.attention_mask = attention_mask[:, -src_seq_len-tgt_seq_len:-tgt_seq_len, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.dtime_seqs_transformed = dtime_seqs_transformed[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.dtime_seqs = self.dtime_transform.inv(self.dtime_seqs_transformed)
        self.batch_non_pad_mask = batch_non_pad_mask[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]

    
class TargetSequence():
    # the data sequence is divided into two parts: history and target
    # [:, -self.tgt_seq_len:] is the target sequence with the length of self.tgt_seq_len
    # target sequence consists of only departure events type = max_type
    # in departure events, dtime is not the time since last event, it is the packet delay
    # [:, -2-self.src_seq_len:-self.tgt_seq_len] is the history sequence with the length: self.src_seq_len
    def __init__(self, batch, src_seq_len, tgt_seq_len, dtime_transform):
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.dtime_transform = dtime_transform
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.tgt_seq_len events in the target
        self.dtime_seqs_transformed = dtime_seqs_transformed[:, -self.tgt_seq_len:]
        self.dtime_seqs = self.dtime_transform.inv(self.dtime_seqs_transformed)
        self.attention_mask = attention_mask[:, -self.tgt_seq_len:, -self.tgt_seq_len:]
        self.batch_non_pad_mask = batch_non_pad_mask[:, -self.tgt_seq_len:]


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Creates a causal (subsequent) mask of shape [size, size]
    where True means "mask out" (cannot attend).
    """
    # upper-triangular (excluding diagonal) -> 1 for positions to mask
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask  # shape [size, size]

def build_decoder_mask(pad_mask: torch.Tensor):
    """
    pad_mask: [batch_size, tgt_seq_len] (1=valid, 0=pad)
    Returns:
        combined_mask: [batch_size, tgt_seq_len, tgt_seq_len] (boolean)
                       True -> "mask out" those positions
    """
    batch_size, seq_len = pad_mask.size()
    # 1) Expand pad_mask into 2D: [batch_size, seq_len] -> [batch_size, 1, seq_len]
    pad_mask_2d = pad_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
    # 2) subsequent_mask for each sample: [seq_len, seq_len]
    sub_mask = subsequent_mask(seq_len).to(pad_mask.device)  # [seq_len, seq_len]
    sub_mask = sub_mask.unsqueeze(0)  # [1, seq_len, seq_len], broadcast later

    # 3) Combine: a position is masked if (it is padded) or (it is beyond the current time).
    # We'll create a broadcast shape [batch_size, seq_len, seq_len].
    #   pad_mask_2d => [batch_size, 1, seq_len]
    #   sub_mask     => [1, seq_len, seq_len]

    # We want to return a boolean mask where True indicates "MASK OUT".
    # We'll invert pad_mask_2d so 1 => "valid" => no mask, 0 => "pad" => mask=True
    pad_mask_2d_inverted = (pad_mask_2d == 0)  # True where pad => do mask
    # Now broadcast them:
    combined_mask = pad_mask_2d_inverted | sub_mask  # logical OR
    # shape [batch_size, seq_len, seq_len]
    return combined_mask