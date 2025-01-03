import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise


class SingleStepArrival(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(SingleStepArrival, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        self.mean_len = model_config.model_specs.get("mean_len", 0.0)
        self.std_len = model_config.model_specs.get("std_len", 1.0)
        self.len_hist_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)

        # Noise regularization, only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)
        if model_config.noise_regularization.event_type['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to length with std dev: {model_config.noise_regularization.event_type['std_dev']}")
            self.nr_len = AddGaussianNoise(mean=0, std=model_config.noise_regularization.event_type['std_dev'], device=self.device)
        else:
            self.nr_len = AddGaussianNoise(mean=0, std=0, device=self.device)

        self.concat_embeddings = model_config.model_specs['embeddings']['concat']

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.mdn_2d = model_config.model_specs['mdn']['2d']
        self.his_len = model_config.model_specs['history']['length']

        if self.mdn_2d:
            self.num_mix_components_2d = model_config.model_specs['mdn']['num_mix_components_2d']
        else:
            self.num_mix_components_dtime = model_config.model_specs['mdn']['num_mix_components_dtime']
            self.num_mix_components_len = model_config.model_specs['mdn']['num_mix_components_len']
        
        self.len_emb_dim = model_config.model_specs['embeddings']['len_emb_dim'] if self.concat_embeddings else self.d_model

        if self.concat_embeddings:
            # size of time embedding is self.d_model minues the total size of the other embeddings
            self.time_emb_size = self.d_model - (int(self.include_len)*self.len_emb_dim)
        else:
            self.time_emb_size = self.d_model
        
        self.use_norm = model_config.use_ln
        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        # Embedding layers defenitions
        # temporal encoding
        self.layer_temporal_encoding = TimePositionalEncoding(
            self.time_emb_size, device=self.device
        )
        # length in bytes encoding (continuous)
        self.layer_len_emb = nn.Linear(
            1, 
            self.len_emb_dim, 
            device=self.device
        )

        # MLP layer (self.feed_forward) without condition
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        # Transformer layers (self.stack_layers)
        self.stack_layers = nn.ModuleList(
            [EncoderLayer(
                self.d_model,
                MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                   output_linear=False),
                use_residual=False,
                feed_forward=self.feed_forward,
                dropout=self.dropout
            ) for _ in range(self.n_layers)])
        
        # prediction linear layers
        if self.mdn_2d:
            self.linear = nn.Linear(self.d_model, 6 * self.num_mix_components_2d)
        else:
            self.dtime_linear = nn.Linear(self.d_model, 3 * self.num_mix_components_dtime)
            self.len_linear = nn.Linear(self.d_model, 3 * self.num_mix_components_len)

    def forward(self, len_seqs_transformed, time_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.
        Returns:
            tensor: hidden states at event times.
        """
        # only linear ones need unsqueeze
        len_seqs = self.len_hist_transform.inv(len_seqs_transformed) # apply inverse transform to len
        len_seqs = len_seqs.float().unsqueeze(-1)

        # [batch_size, seq_len, hidden_size (d_model)]
        # Temporal and type encoding
        time_enc = self.layer_temporal_encoding(time_seqs)

        # 2) Build a list to concatenate later (maybe)
        len_enc = self.layer_len_emb(len_seqs)
        emb_list = [time_enc,len_enc]

        
        if self.concat_embeddings:
            # 3) Concatenate along the last dimension
            # shape -> [B, S, sum_of_emb_dims]
            enc_output = torch.cat(emb_list, dim=-1)

            # [batch_size, seq_len, hidden_size]
            for enc_layer in self.stack_layers:
                #enc_output += time_enc
                enc_output = enc_layer(
                    enc_output,
                    mask=attention_mask
                )
        else:
            enc_output = len_enc
            # [batch_size, seq_len, hidden_size]
            for enc_layer in self.stack_layers:
                enc_output += time_enc
                enc_output = enc_layer(
                    enc_output,
                    mask=attention_mask
                )

        return enc_output


    def joint_distribution(self, enc_out) -> NormalMixtureDistribution2D:
        """Compute the distribution of delta time.

        Args:
            enc_out (tensor): [batch_size, seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution2D: 2d distribution.
        """
        # [batch_size, seq_len, 6 * num_mix_components_2d]
        raw_params = self.linear(enc_out)

        # Extract locs, log_scales, and log_weights from raw_params
        # locs: Means in 2D, so [batch_size, seq_len, num_components_2d, 2]
        locs = raw_params[..., :self.num_mix_components_2d * 2].reshape(raw_params.shape[0], raw_params.shape[1], self.num_mix_components_2d, 2)
        # log_scales: Diagonal and off-diagonal elements of the covariance matrix
        # [batch_size, seq_len, num_components_2d, 3]
        # 3 values per component in 2D: [variance_x, variance_y, covariance_xy]
        log_scales = raw_params[..., self.num_mix_components_2d * 2: self.num_mix_components_2d * 2 + self.num_mix_components_2d * 3].reshape(raw_params.shape[0], raw_params.shape[1], self.num_mix_components_2d, 3)
        # log_weights: Unchanged, normalized weights for each component
        log_weights = raw_params[..., (self.num_mix_components_2d * 2 + self.num_mix_components_2d * 3):]
        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        pred_dist = NormalMixtureDistribution2D(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_dtime=self.mean_dtime,
            std_dtime=self.std_dtime,
            mean_len=self.mean_len,
            std_len=self.std_len,
            device=self.device
        )
        return pred_dist

    def dtime_distribution(self, enc_out) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            enc_out (tensor): [batch_size, seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution.
        """
        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.dtime_linear(enc_out)

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
    
    def len_distribution(self, enc_out) -> NormalMixtureDistribution:
        """Compute the distribution of length.

        Args:
            enc_out (tensor): [batch_size, seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution: length distribution.
        """
        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.len_linear(enc_out)

        locs = raw_params[..., :self.num_mix_components_len]
        log_scales = raw_params[..., self.num_mix_components_len: (2 * self.num_mix_components_len)]
        log_weights = raw_params[..., (2 * self.num_mix_components_len):]
        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        pred_len_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_val=self.mean_len,
            std_val=self.std_len
        )
        return pred_len_dist


    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        # note len_seqs_transformed is type_seqs in dataloader
        time_seqs, dtime_seqs_transformed, len_seqs_transformed, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            len_seqs_transformed[:, :-1], 
            time_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        len_seqs_transformed = len_seqs_transformed.long()

        # select only the last output of the encoder
        enc_out = enc_out[:, -1:, :]

        # loss calculation
        # 1) bring label data
        label_dtimes = dtime_seqs_transformed[:, -1:]
        label_dtimes = self.nr_dtime(label_dtimes) # add noise
        label_lens = len_seqs_transformed[:, -1:]
        label_lens = self.nr_len(label_lens) # add noise
        # stack and reshape to [batch_size, seq_len-1, 2]
        label_2d_data = torch.stack((label_dtimes, label_lens), dim=-1)
        
        # 2) prepare the mask
        event_mask = batch_non_pad_mask[:, -1:]

        # 3) compute the distributions
        if self.mdn_2d:
            # [batch_size, seq_len, 3 * num_mix_components]
            pred_joint_dist = self.joint_distribution(enc_out)

            joint_ll = pred_joint_dist.log_prob(label_2d_data) * event_mask
            joint_loss = -joint_ll.sum()

            # Marginal or conditional log-likelihoods for dtime and len
            # Assuming pred_joint_dist allows marginalization
            pred_dtime_dist = pred_joint_dist.marginalize(0)  # Marginalize over 'len'
            pred_len_dist = pred_joint_dist.marginalize(1)    # Marginalize over 'dtime'

            # Individual log-probabilities
            dtime_ll = pred_dtime_dist.log_prob(label_dtimes) * event_mask
            len_ll = pred_len_dist.log_prob(label_lens) * event_mask

            # Individual losses
            dtime_loss = -dtime_ll.sum()
            len_loss = -len_ll.sum()
        else:
            # dtime loss distributions
            # [batch_size, seq_len, 3 * num_mix_components]
            pred_dtime_dist = self.dtime_distribution(enc_out)
            # length loss distributions
            # [batch_size, seq_len, 3 * num_mix_components]
            pred_len_dist = self.len_distribution(enc_out)

            dtime_ll = pred_dtime_dist.log_prob(label_dtimes) * event_mask
            dtime_loss = -dtime_ll.sum()   

            len_ll = pred_len_dist.log_prob(label_lens) * event_mask
            len_loss = -len_ll.sum()

            joint_ll = dtime_ll + len_ll
            joint_loss = -joint_ll.sum()
        
        num_events = event_mask.sum().item()
        return joint_loss, num_events, dtime_loss, len_loss


    def predict_mean_variance(self, batch, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs_transformed, len_seqs_transformed, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            len_seqs_transformed[:, :-1], 
            time_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        # select the last output
        # [batch_size, 1, 3 * num_mix_components]
        enc_out = enc_out[:, -1:, :]
        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        if not self.mdn_2d:
            pred_dtime_dist = self.dtime_distribution(enc_out)
            pred_dtime = pred_dtime_dist.mean[..., 0]
            pred_dtime_var = pred_dtime_dist.variance[..., 0]

            pred_len_dist = self.len_distribution(enc_out)
            pred_len = pred_len_dist.mean[..., 0]
            pred_len_var = pred_len_dist.variance[..., 0]

        else:
            pred_joint_dist = self.joint_distribution(enc_out)
            mean_pred = pred_joint_dist.mean
            pred_dtime = mean_pred[..., 0]
            pred_len = mean_pred[..., 1]

            # Marginal or conditional log-likelihoods for dtime and len
            # Assuming pred_joint_dist allows marginalization
            pred_dtime_dist = pred_joint_dist.marginalize(0)  # Marginalize over 'len'
            pred_dtime_var = pred_dtime_dist.variance
            pred_len_dist = pred_joint_dist.marginalize(1)    # Marginalize over 'dtime'
            pred_len_var = pred_len_dist.variance


        dtime_label = dtime_seqs_transformed[:, -1]
        len_label = len_seqs_transformed[:, -1]

        return (pred_dtime,pred_dtime_var), (pred_len,pred_len_var), (dtime_label, len_label), event_mask[...,0], num_events


    def predict_probabilities(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs_transformed, len_seqs_transformed, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            len_seqs_transformed[:, :-1], 
            time_seqs[:, :-1], 
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

            return (dtimes_logprob_pred, lens_logprob_pred), (dtime_seqs_transformed, time_seqs, len_seqs_transformed), event_mask[...,0], num_events

        else:
            pred_joint_dist = self.joint_distribution(enc_out)
            
            # Step 1: Create 1D linspace for each dimension
            sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
            sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
            num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
            sample_len_min = prediction_config['probability_generation']['sample_event_type_min']
            sample_len_max = prediction_config['probability_generation']['sample_event_type_max']
            num_steps_len = prediction_config['probability_generation']['num_steps_event_type']
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

            return (joint_logpdfs_pred,), (dtime_seqs_transformed, time_seqs, len_seqs_transformed), event_mask[...,0], num_events
    

    def generate_samples(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs_transformed, len_seqs_transformed, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            len_seqs_transformed[:, :-1], 
            time_seqs[:, :-1], 
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
        len_samples = pred_len_dist.sample((prediction_config['num_samples_event_type'],))

        return (dtimes_samples, len_samples), (dtime_seqs_transformed, time_seqs, len_seqs_transformed), event_mask[...,0], num_events