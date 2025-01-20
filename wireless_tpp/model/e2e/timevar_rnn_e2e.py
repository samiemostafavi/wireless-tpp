import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Sequence():
    def __init__(self, batch, device, dtime_transform, len_transform, interarrival_time_transform):
        self.device = device
        self.dtime_transform = dtime_transform
        self.len_transform = len_transform
        self.interarrival_time_transform = interarrival_time_transform
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, interarrival_time_seqs_transformed, label_mask_seqs, non_pad_mask, attention_mask = batch

        self.dtime_seqs_transformed = dtime_seqs_transformed
        self.dtime_seqs = self.dtime_transform.inv(self.dtime_seqs_transformed)
        self.interarrival_time_seqs_transformed = interarrival_time_seqs_transformed
        self.interarrival_time_seqs = self.interarrival_time_transform.inv(self.interarrival_time_seqs_transformed)
        self.len_seqs_transformed = len_seqs_transformed
        self.len_seqs = self.len_transform.inv(self.len_seqs_transformed)
        self.non_pad_mask = non_pad_mask
        self.attention_mask = attention_mask
        self.slot_seqs = slot_seqs
        self.mcs_seqs = mcs_seqs
        self.mretx_seqs = mretx_seqs
        self.rfailed_seqs = rfailed_seqs
        self.num_rbs_seqs = num_rbs_seqs
        self.time_seqs = time_seqs
        self.type_seqs = type_seqs
        self.label_mask_seqs = label_mask_seqs

    def get_all(self):
        return self.slot_seqs, self.len_seqs, self.len_seqs_transformed, self.mcs_seqs, self.mretx_seqs, self.rfailed_seqs, self.num_rbs_seqs, self.time_seqs, self.dtime_seqs, self.dtime_seqs_transformed, self.type_seqs, self.interarrival_time_seqs, self.interarrival_time_seqs_transformed, self.label_mask_seqs, self.non_pad_mask, self.attention_mask



class TimeVarRecurrentE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(TimeVarRecurrentE2E, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        self.mean_len = model_config.model_specs.get("mean_len", 0.0)
        self.std_len = model_config.model_specs.get("std_len", 1.0)
        self.mean_interarrival_time = model_config.model_specs.get("mean_interarrival_time", 0.0)
        self.std_interarrival_time = model_config.model_specs.get("std_interarrival_time", 1.0)
        logger.info(f"TimeVarRecurrentE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)
        logger.info(f"TimeVarRecurrentE2E loading mean and std of len: {self.mean_len}, {self.std_len}")
        self.len_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)
        logger.info(f"TimeVarRecurrentE2E loading mean and std of interarrival time: {self.mean_interarrival_time}, {self.std_interarrival_time}")
        self.interarrival_time_transform = D.AffineTransform(loc=self.mean_interarrival_time, scale=self.std_interarrival_time)

        # Noise regularization, only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)

        assert model_config.model_specs['rnn_type'] == 'lstm' or model_config.model_specs['rnn_type'] == 'gru' or model_config.model_specs['rnn_type'] == 'rnn'
        self.rnn_type = model_config.model_specs['rnn_type']
        self.num_layers = model_config.model_specs['num_layers']
        self.teacher_forcing = model_config.model_specs['teacher_forcing']
        self.bidirectional = bool(model_config.model_specs['bidirectional'])
        logger.info(f"RNN type: {self.rnn_type}, num_layers: {self.num_layers}, bidirectional: {self.bidirectional}")

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate

        self.num_mix_components = model_config.model_specs['mdn']['num_mix_components']
        
        self.include_dtime_embedding = model_config.model_specs['embeddings']['include_dtime']
        self.dtime_emb_dim = self.d_model

        self.include_time_embedding = model_config.model_specs['embeddings']['include_time']
        self.time_emb_dim = self.d_model

        self.include_interarrival_time_embedding = model_config.model_specs['embeddings']['include_interarrival_time']
        self.interarrival_time_emb_dim = self.d_model

        self.include_len = model_config.model_specs['embeddings']['include_len']
        self.len_emb_dim = self.d_model
        
        self.include_slot = model_config.model_specs['embeddings']['include_slot']
        self.slot_emb_dim = self.d_model
        
        self.include_mcs = model_config.model_specs['embeddings']['include_mcs']
        self.mcs_emb_dim = self.d_model
        
        self.include_mretx = model_config.model_specs['embeddings']['include_mretx']
        self.mretx_emb_dim = self.d_model

        self.include_rfailed = model_config.model_specs['embeddings']['include_rfailed']
        self.rfailed_emb_dim = self.d_model

        self.time_emb_size = self.d_model
        self.PAD_TOKEN = -1.0

        # History embedding configurations
        # slots embedding
        self.num_slots_types = 21  # slot indices: 0 to 19 (20 types), and padding token
        self.slots_pad_id = 20
        # mcs embedding
        self.num_mcs_types = 30  # MCS indices: 0 to 28 (29 types), and padding token
        self.mcs_pad_id = 29
        # retransmissions embedding
        self.num_mretx_types = 5  # retransmission indices: 0 to 3 (4 types), and padding token
        self.mretx_pad_id = 4
        # rlc failed embedding
        self.num_rfailed_types = 3  # failed attempt indices: 0 and 1 (2 types), and padding token
        self.rfailed_pad_id = 2
        # rum rbs embedding
        self.num_rbs_types = 107  # number of rbs 0-106 (107 types), and padding token
        self.rbs_pad_id = 106

        # Embedding layers defenitions
        # delay embedding layer
        self.dtime_emb_layer = nn.Linear(1, self.d_model)
        if self.include_time_embedding:
            self.layer_time_embedding = TimePositionalEncoding(
                self.time_emb_dim, device=self.device
            )
        if self.include_interarrival_time_embedding:
            self.layer_interarrival_time_embedding = nn.Linear(
                1, 
                self.interarrival_time_emb_dim, 
                device=self.device
            )
        if self.include_slot:
            # slot number encoding
            self.layer_slot_emb = nn.Embedding(
                self.num_slots_types,
                self.slot_emb_dim,
                padding_idx=self.slots_pad_id,
                device=self.device
            )
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
        if self.include_rfailed:
            # failed attempt encoding
            self.layer_rfailed_emb = nn.Embedding(
                self.num_rfailed_types,
                self.rfailed_emb_dim, 
                padding_idx=self.rfailed_pad_id,
                device=self.device
            )
        if self.include_len:
            # length in bytes encoding (continuous)
            self.layer_len_emb = nn.Linear(
                1, 
                self.len_emb_dim, 
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
            self.linear = nn.Linear(2*self.d_model, 3 * self.num_mix_components)
        else:
            self.linear = nn.Linear(self.d_model, 3 * self.num_mix_components)


    def encode(self, seq_obj : Sequence):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, seq_len], attention masks.
        Returns:
            tensor: hidden states at event times.
        """

        # only linear ones need unsqueeze
        # convert type_seqs to int type for embedding
        slot_seqs = seq_obj.slot_seqs.long()
        mcs_seqs = seq_obj.mcs_seqs.long()
        mretx_seqs = seq_obj.mretx_seqs.long()
        rfailed_seqs = seq_obj.rfailed_seqs.long()
        
        # output should be [batch_size, seq_len, hidden_size (d_model)]
        enc_output = torch.zeros((slot_seqs.shape[0],slot_seqs.shape[1],self.d_model), device=self.device)

        if self.include_dtime_embedding:
            dtime_enc = self.dtime_emb_layer(seq_obj.dtime_seqs.unsqueeze(-1))
            enc_output += dtime_enc

        if self.include_interarrival_time_embedding:
            interarrival_time_enc = self.layer_interarrival_time_embedding(seq_obj.interarrival_time_seqs.unsqueeze(-1))
            enc_output += interarrival_time_enc

        if self.include_time_embedding:
            time_enc = self.layer_time_embedding(seq_obj.time_seqs.unsqueeze(-1))
            enc_output += time_enc

        if self.include_len:
            len_seqs = seq_obj.len_seqs # applied inverse transform to len
            len_seqs = len_seqs.float().unsqueeze(-1)
            len_enc = self.layer_len_emb(len_seqs)
            enc_output += len_enc

        if self.include_slot: 
            slot_enc = self.layer_slot_emb(slot_seqs)
            enc_output += slot_enc
        
        if self.include_mcs:
            mcs_enc = self.layer_mcs_emb(mcs_seqs)
            enc_output += mcs_enc

        if self.include_mretx:
            mretx_enc = self.layer_mretx_emb(mretx_seqs)
            enc_output += mretx_enc

        if self.include_rfailed:
            rfailed_enc = self.layer_rfailed_emb(rfailed_seqs)
            enc_output += rfailed_enc

        return enc_output

    def get_pred_distribution(self, rnn_out) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            rnn_out (tensor): [batch_size, seq_len, d_model(*2 if bidirectional)], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution.
        """
        # input: [batch_size, seq_len, d_model(*2 if bidirectional)]
        # output: [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.linear(rnn_out)

        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]
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

    def loglike_loss(self, batch, forward=True):

        if self.teacher_forcing:
            dtime_loss, num_predictions = self.loglike_loss_tf(batch, forward)
        else:
            dtime_loss, num_predictions = self.loglike_loss_eval(batch)

        return dtime_loss, num_predictions.item(), None, None


    def loglike_loss_tf(self, batch, forward=True):

        # check if we are running validation or training
        if not forward:
            dtime_loss, num_predictions = self.loglike_loss_eval(batch)
            return dtime_loss, num_predictions.item(), None, None

        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embedding = torch.zeros_like(embedding)  # Initialize a zero tensor with the same shape as embedding
        shifted_embedding[:, 1:, :] = embedding[:, :-1, :]  # Shift embeddings to the right

        # Packed input with shifted embeddings
        seq_mask = seq_obj.non_pad_mask.float()  # Mask to handle padding
        seq_lengths = seq_mask.sum(dim=1).long()

        # Pack the shifted embeddings
        packed_input = pack_padded_sequence(shifted_embedding, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.layer_rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)  # [batch_size, seq_len, hidden_size]

        # RNN outputs and loss computation
        pred_dist = self.get_pred_distribution(rnn_out)  # Predict based on RNN output

        # Compute the log-likelihood loss
        assert seq_obj.dtime_seqs_transformed.shape == pred_dist.mean.shape  # Ensure alignment
        dtime_ll = pred_dist.log_prob(seq_obj.dtime_seqs_transformed) * seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()
        dtime_loss = -dtime_ll.sum()

        num_predictions = (seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()).sum()

        return dtime_loss, num_predictions

    def get_rnn_input_step(self, rnn_out_step, interarrival_time):
        # rnn_out: [batch_size, 1, d_model]
        # interarrival_time: [batch_size, 1]
        pred_dist_step = self.get_pred_distribution(rnn_out_step)
        pred_dtime_step = pred_dist_step.mean
        pred_dtime_step_transformed = self.dtime_transform.inv(pred_dtime_step)
        enc_step = torch.zeros_like(rnn_out_step, device=self.device)
        if self.include_dtime_embedding:
            dtime_enc = self.dtime_emb_layer(pred_dtime_step_transformed.unsqueeze(-1))
            enc_step += dtime_enc
        if self.include_interarrival_time_embedding:
            interarrival_time_enc = self.layer_interarrival_time_embedding(interarrival_time.unsqueeze(-1))
            enc_step += interarrival_time_enc
        input_step = enc_step
        return input_step

    def loglike_loss_eval(self, batch):

        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embedding = torch.zeros_like(embedding)  # Initialize a zero tensor with the same shape as embedding
        shifted_embedding[:, 1:, :] = embedding[:, :-1, :]  # Shift embeddings to the right

        # history_mask_seqs: [batch_size, seq_len]
        history_mask = (seq_obj.label_mask_seqs == 0).float() * seq_obj.non_pad_mask.float()
        history_seq_lengths = history_mask.sum(dim=1).long()

        # feed in the hisotry data
        packed_input = pack_padded_sequence(shifted_embedding, history_seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, history_hidden = self.layer_rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        prediction_mask = seq_obj.label_mask_seqs.float() * seq_obj.non_pad_mask.float()
        prediction_lengths = prediction_mask.sum(dim=1).long()
 
        # Initialize tensors for predictions
        batch_size, seq_len, d_model = embedding.size()
        predictions = torch.zeros(batch_size, seq_len, d_model, device=self.device)
        num_predictions = 0
        dtime_loss = 0.0

        # Generate predictions for each sequence
        for i in range(batch_size):
            # Get prediction length for this sequence
            pred_len = prediction_lengths[i].item()
            interarrival_time_batch = seq_obj.interarrival_time_seqs_transformed[i,0]
            if pred_len == 0:
                continue  # No predictions for this sequence

            # Extract the last valid history output
            last_history_index = history_seq_lengths[i] - 1
            last_history_rnn_out = rnn_out[i, last_history_index, :].unsqueeze(0).unsqueeze(1)  # [1, 1, d_model]
            # encode it to get the input for the next step
            input_step = self.get_rnn_input_step(last_history_rnn_out, interarrival_time_batch)

            # Initialize hidden state for the prediction loop
            prev_hidden = (history_hidden[0][:, i:i+1, :], history_hidden[1][:, i:i+1, :])  # Extract hidden and cell state

            num_predictions += prediction_lengths[i].item()

            # Predict step-by-step
            for t in range(pred_len):
                output_step, prev_hidden = self.layer_rnn(input_step, prev_hidden)  # RNN step
                # Save prediction
                predictions[i, history_seq_lengths[i] + t, :] = output_step.squeeze(1)  
                # Update input for the next step
                input_step = self.get_rnn_input_step(output_step, interarrival_time_batch)

        # Apply prediction mask to filter out invalid positions
        pred_dist = self.get_pred_distribution(predictions)
        assert seq_obj.dtime_seqs_transformed.shape == pred_dist.mean.shape # [batch_size, seq_len]
        dtime_ll = pred_dist.log_prob(seq_obj.dtime_seqs_transformed) * seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()
        dtime_loss = -dtime_ll.sum()

        num_predictions = (seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()).sum()
        return dtime_loss, num_predictions


    def predict_mean_variance(self, batch):
        """
        Predict mean & variance for `tgt_seq_len` future steps, given `src_seq_len` historical data,
        by auto-regressively feeding the predicted *mean* from each step back as the next input.

        Returns:
            all_means: [batch_size, tgt_seq_len]
            all_vars:  [batch_size, tgt_seq_len]
        """
        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # history_mask_seqs: [batch_size, seq_len]
        history_mask = (seq_obj.label_mask_seqs == 0).float() * seq_obj.non_pad_mask.float()
        history_seq_lengths = history_mask.sum(dim=1).long()

        # feed in the hisotry data
        packed_input = pack_padded_sequence(embedding, history_seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, history_hidden = self.layer_rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        prediction_mask = seq_obj.label_mask_seqs.float() * seq_obj.non_pad_mask.float()
        prediction_lengths = prediction_mask.sum(dim=1).long()
 
        # Initialize tensors for predictions
        batch_size, seq_len, d_model = embedding.size()
        predictions = torch.zeros(batch_size, seq_len, d_model, device=self.device)
        num_predictions = 0
        loss = 0.0

        # Generate predictions for each sequence
        for i in range(batch_size):
            # Get prediction length for this sequence
            pred_len = prediction_lengths[i].item()
            interarrival_time_batch = seq_obj.interarrival_time_seqs_transformed[i,0]
            if pred_len == 0:
                continue  # No predictions for this sequence

            # Initialize the prediction loop
            prev_hidden = (history_hidden[0][:, i:i+1, :], history_hidden[1][:, i:i+1, :])  # Extract hidden and cell state
            input_step = embedding[i, history_seq_lengths[i] - 1, :].unsqueeze(0).unsqueeze(1)  # Last step of history

            num_predictions += prediction_lengths[i].item()

            # Predict step-by-step
            for t in range(pred_len):
                output_step, prev_hidden = self.layer_rnn(input_step, prev_hidden)  # RNN step
                predictions[i, history_seq_lengths[i] + t, :] = output_step.squeeze(1)  # Save prediction

                pred_dist_step = self.get_pred_distribution(output_step)
                pred_dtime_step = pred_dist_step.mean
                pred_dtime_step_transformed = self.dtime_transform.inv(pred_dtime_step)

                # Update input for the next step
                enc_last_pred = torch.zeros((1,1,self.d_model), device=self.device)
                if self.include_dtime_embedding:
                    dtime_enc = self.dtime_emb_layer(pred_dtime_step_transformed.unsqueeze(-1))
                    enc_last_pred += dtime_enc

                if self.include_interarrival_time_embedding:
                    interarrival_time_enc = self.layer_interarrival_time_embedding(interarrival_time_batch.unsqueeze(-1))
                    enc_last_pred += interarrival_time_enc
                
                input_step = enc_last_pred

        # Apply prediction mask to filter out invalid positions
        pred_dist = self.get_pred_distribution(predictions)
        pred_dtime = pred_dist.mean
        pred_dtime_var = pred_dist.variance

        return (pred_dtime,pred_dtime_var), (None,None), (seq_obj.dtime_seqs_transformed, None), prediction_mask, num_predictions


class TimeVarHalfRecurrentE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(TimeVarHalfRecurrentE2E, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        self.mean_len = model_config.model_specs.get("mean_len", 0.0)
        self.std_len = model_config.model_specs.get("std_len", 1.0)
        self.mean_interarrival_time = model_config.model_specs.get("mean_interarrival_time", 0.0)
        self.std_interarrival_time = model_config.model_specs.get("std_interarrival_time", 1.0)
        logger.info(f"TimeVarRecurrentE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)
        logger.info(f"TimeVarRecurrentE2E loading mean and std of len: {self.mean_len}, {self.std_len}")
        self.len_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)
        logger.info(f"TimeVarRecurrentE2E loading mean and std of interarrival time: {self.mean_interarrival_time}, {self.std_interarrival_time}")
        self.interarrival_time_transform = D.AffineTransform(loc=self.mean_interarrival_time, scale=self.std_interarrival_time)

        # Noise regularization, only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)

        assert model_config.model_specs['rnn_type'] == 'lstm' or model_config.model_specs['rnn_type'] == 'gru' or model_config.model_specs['rnn_type'] == 'rnn'
        self.rnn_type = model_config.model_specs['rnn_type']
        self.num_layers = model_config.model_specs['num_layers']
        self.teacher_forcing = model_config.model_specs['teacher_forcing']
        self.bidirectional = bool(model_config.model_specs['bidirectional'])
        logger.info(f"RNN type: {self.rnn_type}, num_layers: {self.num_layers}, bidirectional: {self.bidirectional}")

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate

        self.num_mix_components = model_config.model_specs['mdn']['num_mix_components']
        self.max_tgt_seq_len = model_config.model_specs['max_tgt_seq_len']
        self.max_src_seq_len = model_config.model_specs['max_src_seq_len']

        self.include_dtime_embedding = model_config.model_specs['embeddings']['include_dtime']
        self.dtime_emb_dim = self.d_model

        self.include_time_embedding = model_config.model_specs['embeddings']['include_time']
        self.time_emb_dim = self.d_model

        self.include_interarrival_time_embedding = model_config.model_specs['embeddings']['include_interarrival_time']
        self.interarrival_time_emb_dim = self.d_model

        self.include_len = model_config.model_specs['embeddings']['include_len']
        self.len_emb_dim = self.d_model
        
        self.include_slot = model_config.model_specs['embeddings']['include_slot']
        self.slot_emb_dim = self.d_model
        
        self.include_mcs = model_config.model_specs['embeddings']['include_mcs']
        self.mcs_emb_dim = self.d_model
        
        self.include_mretx = model_config.model_specs['embeddings']['include_mretx']
        self.mretx_emb_dim = self.d_model

        self.include_rfailed = model_config.model_specs['embeddings']['include_rfailed']
        self.rfailed_emb_dim = self.d_model

        self.time_emb_size = self.d_model
        self.PAD_TOKEN = -1.0

        # History embedding configurations
        # slots embedding
        self.num_slots_types = 21  # slot indices: 0 to 19 (20 types), and padding token
        self.slots_pad_id = 20
        # mcs embedding
        self.num_mcs_types = 30  # MCS indices: 0 to 28 (29 types), and padding token
        self.mcs_pad_id = 29
        # retransmissions embedding
        self.num_mretx_types = 5  # retransmission indices: 0 to 3 (4 types), and padding token
        self.mretx_pad_id = 4
        # rlc failed embedding
        self.num_rfailed_types = 3  # failed attempt indices: 0 and 1 (2 types), and padding token
        self.rfailed_pad_id = 2
        # rum rbs embedding
        self.num_rbs_types = 107  # number of rbs 0-106 (107 types), and padding token
        self.rbs_pad_id = 106

        # Embedding layers defenitions
        # delay embedding layer
        self.dtime_emb_layer = nn.Linear(1, self.d_model)
        if self.include_time_embedding:
            self.layer_time_embedding = TimePositionalEncoding(
                self.time_emb_dim, device=self.device
            )
        if self.include_interarrival_time_embedding:
            self.layer_interarrival_time_embedding = nn.Linear(
                1, 
                self.interarrival_time_emb_dim, 
                device=self.device
            )
        if self.include_slot:
            # slot number encoding
            self.layer_slot_emb = nn.Embedding(
                self.num_slots_types,
                self.slot_emb_dim,
                padding_idx=self.slots_pad_id,
                device=self.device
            )
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
        if self.include_rfailed:
            # failed attempt encoding
            self.layer_rfailed_emb = nn.Embedding(
                self.num_rfailed_types,
                self.rfailed_emb_dim, 
                padding_idx=self.rfailed_pad_id,
                device=self.device
            )
        if self.include_len:
            # length in bytes encoding (continuous)
            self.layer_len_emb = nn.Linear(
                1, 
                self.len_emb_dim, 
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
        raw_params_dim = self.max_tgt_seq_len*3*self.num_mix_components
        if self.bidirectional:
            input_dim = 2*self.d_model
        else:
            input_dim = self.d_model
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, raw_params_dim)
        )


    def encode(self, seq_obj : Sequence):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, seq_len], attention masks.
        Returns:
            tensor: hidden states at event times.
        """

        # only linear ones need unsqueeze
        # convert type_seqs to int type for embedding
        slot_seqs = seq_obj.slot_seqs.long()
        mcs_seqs = seq_obj.mcs_seqs.long()
        mretx_seqs = seq_obj.mretx_seqs.long()
        rfailed_seqs = seq_obj.rfailed_seqs.long()
        
        # output should be [batch_size, seq_len, hidden_size (d_model)]
        enc_output = torch.zeros((slot_seqs.shape[0],slot_seqs.shape[1],self.d_model), device=self.device)

        if self.include_dtime_embedding:
            dtime_enc = self.dtime_emb_layer(seq_obj.dtime_seqs.unsqueeze(-1))
            enc_output += dtime_enc

        if self.include_interarrival_time_embedding:
            interarrival_time_enc = self.layer_interarrival_time_embedding(seq_obj.interarrival_time_seqs.unsqueeze(-1))
            enc_output += interarrival_time_enc

        if self.include_time_embedding:
            time_enc = self.layer_time_embedding(seq_obj.time_seqs.unsqueeze(-1))
            enc_output += time_enc

        if self.include_len:
            len_seqs = seq_obj.len_seqs # applied inverse transform to len
            len_seqs = len_seqs.float().unsqueeze(-1)
            len_enc = self.layer_len_emb(len_seqs)
            enc_output += len_enc

        if self.include_slot: 
            slot_enc = self.layer_slot_emb(slot_seqs)
            enc_output += slot_enc
        
        if self.include_mcs:
            mcs_enc = self.layer_mcs_emb(mcs_seqs)
            enc_output += mcs_enc

        if self.include_mretx:
            mretx_enc = self.layer_mretx_emb(mretx_seqs)
            enc_output += mretx_enc

        if self.include_rfailed:
            rfailed_enc = self.layer_rfailed_emb(rfailed_seqs)
            enc_output += rfailed_enc

        return enc_output

    def get_pred_distribution(self, rnn_out) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            rnn_out (tensor): [batch_size, d_model(*2 if bidirectional)], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution [batch_size, self.max_tgt_seq_len]
        """
        # input: [batch_size, d_model(*2 if bidirectional)]
        # output: [batch_size, self.max_tgt_seq_len * 3 * num_mix_components]
        raw_params = self.linear(rnn_out)

        # convert raw_params dims to [batch_size, self.max_tgt_seq_len, 3 * num_mix_components]
        raw_params = raw_params.view(-1, self.max_tgt_seq_len, 3 * self.num_mix_components)

        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]
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

    def loglike_loss(self, batch, forward=True):

        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # history_mask_seqs: [batch_size, seq_len]
        history_mask = (seq_obj.label_mask_seqs == 0).float() * seq_obj.non_pad_mask.float()

        # zero out non-history positions
        embedding = embedding*history_mask.unsqueeze(-1)

        # feed the embeddings
        rnn_out, _ = self.layer_rnn(embedding)

        # RNN outputs and loss computation
        # Predict based on the last history output (there could be some padding)
        # here we assume padding side is right
        pred_dist = self.get_pred_distribution(rnn_out[:,self.max_src_seq_len,:])  

        # create a new mask for the prediction
        pred_mask = seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long() # [batch_size, seq_len]
        batch_size = pred_mask.shape[0]
        history_lengths = history_mask.sum(dim=1).long() # [batch_size]
        pred_lengths = pred_mask.sum(dim=1).long() # [batch_size]
        dtime_loss = 0
        for i in range(batch_size):
            pred_length = pred_lengths[i].item()
            history_length = history_lengths[i].item()
            # Compute the log-likelihood loss
            dtime_ll = pred_dist.log_prob(
                seq_obj.dtime_seqs_transformed[i, history_length:history_length+self.max_tgt_seq_len]
            ) * pred_mask[i, history_length:history_length+self.max_tgt_seq_len]
            dtime_loss += -dtime_ll.sum()

        num_predictions = (seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()).sum()

        return dtime_loss, num_predictions.item(), None, None

    def predict_mean_variance(self, batch):
        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # history_mask_seqs: [batch_size, seq_len]
        history_mask = (seq_obj.label_mask_seqs == 0).float() * seq_obj.non_pad_mask.float()

        # zero out non-history positions
        embedding = embedding*history_mask.unsqueeze(-1)

        # feed the embeddings
        rnn_out, _ = self.layer_rnn(embedding)

        # RNN outputs and loss computation
        # Predict based on the last history output (there could be some padding)
        pred_dist = self.get_pred_distribution(rnn_out[:,self.max_src_seq_len,:])

        # create a new mask for the prediction
        pred_mask = seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long() # [batch_size, seq_len]
        batch_size = pred_mask.shape[0]
        seq_len = pred_mask.shape[1]
        history_lengths = history_mask.sum(dim=1).long() # [batch_size]
        pred_lengths = pred_mask.sum(dim=1).long() # [batch_size]
        pred_dtime = torch.zeros((batch_size, seq_len), device=self.device)
        pred_dtime_var = torch.zeros((batch_size, seq_len), device=self.device)
        for i in range(batch_size):
            pred_length = pred_lengths[i].item()
            history_length = history_lengths[i].item()
            # Compute the log-likelihood loss
            pred_dtime[i, history_length:history_length+pred_length] = pred_dist.mean[i,:] * pred_mask[i, history_length:history_length+self.max_tgt_seq_len]
            pred_dtime_var[i, history_length:history_length+pred_length] = pred_dist.variance[i,:] * pred_mask[i, history_length:history_length+self.max_tgt_seq_len]

        num_predictions = (seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()).sum()
        return (pred_dtime,pred_dtime_var), (None,None), (seq_obj.dtime_seqs_transformed, None), pred_mask, num_predictions


class SequenceSeperate():
    def __init__(self, batch, device, src_seq_len, tgt_seq_len, dtime_transform, len_transform, interarrival_time_transform):
        self.device = device
        self.dtime_transform = dtime_transform
        self.len_transform = len_transform
        self.interarrival_time_transform = interarrival_time_transform
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, interarrival_time_seqs_transformed, non_pad_mask, attention_mask = batch

        # only consider the last self.tgt_seq_len events in the target
        self.dtime_seqs_transformed = dtime_seqs_transformed
        self.dtime_seqs = self.dtime_transform.inv(self.dtime_seqs_transformed)
        self.interarrival_time_seqs_transformed = interarrival_time_seqs_transformed
        self.interarrival_time_seqs = self.interarrival_time_transform.inv(self.interarrival_time_seqs_transformed)
        self.len_seqs_transformed = len_seqs_transformed
        self.len_seqs = self.len_transform.inv(self.len_seqs_transformed)
        self.non_pad_mask = non_pad_mask
        self.attention_mask = attention_mask
        self.slot_seqs = slot_seqs
        self.mcs_seqs = mcs_seqs
        self.mretx_seqs = mretx_seqs
        self.rfailed_seqs = rfailed_seqs
        self.num_rbs_seqs = num_rbs_seqs
        self.time_seqs = time_seqs
        self.type_seqs = type_seqs

        self.src_slot_seqs = self.slot_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_len_seqs = self.len_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_len_seqs_transformed = self.len_seqs_transformed[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_mcs_seqs = self.mcs_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_mretx_seqs = self.mretx_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_rfailed_seqs = self.rfailed_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_num_rbs_seqs = self.num_rbs_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_time_seqs = self.time_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_dtime_seqs = self.dtime_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_dtime_seqs_transformed = self.dtime_seqs_transformed[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_type_seqs = self.type_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_interarrival_time_seqs = self.interarrival_time_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_interarrival_time_seqs_transformed = self.interarrival_time_seqs_transformed[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_non_pad_mask = self.non_pad_mask[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.src_attention_mask = self.attention_mask[:, -src_seq_len-tgt_seq_len:-tgt_seq_len, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        

        self.tgt_slot_seqs = self.slot_seqs[:, -tgt_seq_len:]
        self.tgt_len_seqs = self.len_seqs[:, -tgt_seq_len:]
        self.tgt_len_seqs_transformed = self.len_seqs_transformed[:, -tgt_seq_len:]
        self.tgt_mcs_seqs = self.mcs_seqs[:, -tgt_seq_len:]
        self.tgt_mretx_seqs = self.mretx_seqs[:, -tgt_seq_len:]
        self.tgt_rfailed_seqs = self.rfailed_seqs[:, -tgt_seq_len:]
        self.tgt_num_rbs_seqs = self.num_rbs_seqs[:, -tgt_seq_len:]
        self.tgt_time_seqs = self.time_seqs[:, -tgt_seq_len:]
        self.tgt_dtime_seqs = self.dtime_seqs[:, -tgt_seq_len:]
        self.tgt_dtime_seqs_transformed = self.dtime_seqs_transformed[:, -tgt_seq_len:]
        self.tgt_type_seqs = self.type_seqs[:, -tgt_seq_len:]
        self.tgt_interarrival_time_seqs = self.interarrival_time_seqs[:, -tgt_seq_len:]
        self.tgt_interarrival_time_seqs_transformed = self.interarrival_time_seqs_transformed[:, -tgt_seq_len:]
        self.tgt_non_pad_mask = self.non_pad_mask[:, -tgt_seq_len:]
        self.tgt_attention_mask = self.attention_mask[:, -tgt_seq_len:, -tgt_seq_len:]

    def get_all(self):
        return self.slot_seqs, self.len_seqs, self.len_seqs_transformed, self.mcs_seqs, self.mretx_seqs, self.rfailed_seqs, self.num_rbs_seqs, self.time_seqs, self.dtime_seqs, self.dtime_seqs_transformed, self.type_seqs, self.interarrival_time_seqs, self.interarrival_time_seqs_transformed, self.non_pad_mask, self.attention_mask
    
    def get_src_seqs(self):
        return self.src_slot_seqs, self.src_len_seqs, self.src_len_seqs_transformed, self.src_mcs_seqs, self.src_mretx_seqs, self.src_rfailed_seqs, self.src_num_rbs_seqs, self.src_time_seqs, self.src_dtime_seqs, self.src_dtime_seqs_transformed, self.src_type_seqs, self.src_interarrival_time_seqs, self.src_interarrival_time_seqs_transformed, self.src_non_pad_mask, self.src_attention_mask

    def get_target_seqs(self):
        return self.tgt_slot_seqs, self.tgt_len_seqs, self.tgt_len_seqs_transformed, self.tgt_mcs_seqs, self.tgt_mretx_seqs, self.tgt_rfailed_seqs, self.tgt_num_rbs_seqs, self.tgt_time_seqs, self.tgt_dtime_seqs, self.tgt_dtime_seqs_transformed, self.tgt_type_seqs, self.tgt_interarrival_time_seqs, self.tgt_interarrival_time_seqs_transformed, self.tgt_non_pad_mask, self.tgt_attention_mask