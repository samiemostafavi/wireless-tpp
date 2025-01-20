import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequenceSeperate():
    def __init__(self, batch, device, src_seq_len, tgt_seq_len, dtime_transform, len_transform, interarrival_time_transform):
        self.device = device
        self.dtime_transform = dtime_transform
        self.len_transform = len_transform
        self.interarrival_time_transform = interarrival_time_transform
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, interarrival_time_seqs_transformed, label_mask_seqs, non_pad_mask, attention_mask = batch

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

    def get_tgt_seqs(self):
        return self.tgt_slot_seqs, self.tgt_len_seqs, self.tgt_len_seqs_transformed, self.tgt_mcs_seqs, self.tgt_mretx_seqs, self.tgt_rfailed_seqs, self.tgt_num_rbs_seqs, self.tgt_time_seqs, self.tgt_dtime_seqs, self.tgt_dtime_seqs_transformed, self.tgt_type_seqs, self.tgt_interarrival_time_seqs, self.tgt_interarrival_time_seqs_transformed, self.tgt_non_pad_mask, self.tgt_attention_mask
    
    def get_element_at_idx(self, idx):
        """
        input: idx (int) from 0 to self.tgt_seq_len+self.src_seq_len
        output: [batch_size, 1]
        """
        assert idx < self.tgt_seq_len+self.src_seq_len
        assert idx >= 0
        return self.slot_seqs[:,idx].unsqueeze(-1), self.len_seqs[:,idx].unsqueeze(-1), self.len_seqs_transformed[:,idx].unsqueeze(-1), self.mcs_seqs[:,idx].unsqueeze(-1), self.mretx_seqs[:,idx].unsqueeze(-1), self.rfailed_seqs[:,idx].unsqueeze(-1), self.num_rbs_seqs[:,idx].unsqueeze(-1), self.time_seqs[:,idx].unsqueeze(-1), self.dtime_seqs[:,idx].unsqueeze(-1), self.dtime_seqs_transformed[:,idx].unsqueeze(-1), self.type_seqs[:,idx].unsqueeze(-1), self.interarrival_time_seqs[:,idx].unsqueeze(-1), self.interarrival_time_seqs_transformed[:,idx].unsqueeze(-1), self.non_pad_mask[:,idx].unsqueeze(-1), self.attention_mask[:,idx].unsqueeze(-1)


class EncoderE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(EncoderE2E, self).__init__(model_config)

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

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate

        self.num_mix_components = model_config.model_specs['mdn']['num_mix_components']
        self.tgt_seq_len = model_config.model_specs['tgt_seq_len']
        self.src_seq_len = model_config.model_specs['src_seq_len']
        self.teacher_forcing = model_config.model_specs['teacher_forcing']
        self.n_encoder_heads = model_config.model_specs['encoder']['num_heads']
        self.n_encoder_layers = model_config.model_specs['encoder']['num_layers']
        self.encoder_use_residual = model_config.model_specs['encoder']['use_residual']

        # dtime embedding MUST EXIST
        self.dtime_emb_dim = self.d_model

        self.include_interarrival_time = model_config.model_specs['embeddings']['include_interarrival_time']
        self.interarrival_time_emb_dim = self.d_model

        self.include_len = model_config.model_specs['embeddings']['include_len']
        self.len_emb_dim = self.d_model

        self.include_slot = model_config.model_specs['embeddings']['include_slot']
        self.slot_emb_dim = self.d_model

        self.include_time_embedding = model_config.model_specs['embeddings']['include_time']
        self.time_emb_dim = self.d_model
        
        self.include_mcs = model_config.model_specs['embeddings']['include_mcs']
        self.include_mcs_in_tgt = model_config.model_specs['target']['include_mcs']
        self.mcs_emb_dim = self.d_model
        
        self.include_mretx = model_config.model_specs['embeddings']['include_mretx']
        self.include_mretx_in_tgt = model_config.model_specs['target']['include_mretx']
        self.mretx_emb_dim = self.d_model

        self.include_rfailed = model_config.model_specs['embeddings']['include_rfailed']
        self.include_rfailed_in_tgt = model_config.model_specs['target']['include_rfailed']
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
        # delay embedding layer (MUST EXIST)
        self.dtime_emb_layer = nn.Linear(1, self.d_model)
        if self.include_time_embedding:
            self.layer_time_embedding = TimePositionalEncoding(
                self.time_emb_dim, device=self.device
            )
        if self.include_interarrival_time:
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

        # encoder MLP layer 
        self.feed_forward_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        # Transformer encoder layers (self.encoder_layers)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(
                d_model=self.d_model,
                self_attn=MultiHeadAttention(self.n_encoder_heads, self.d_model, self.d_model, self.dropout,
                                output_linear=False),
                use_residual=self.encoder_use_residual,
                feed_forward=self.feed_forward_encoder,
                dropout=self.dropout
            ) for _ in range(self.n_encoder_layers)])

        # prediction linear layer
        self.linear = nn.Linear(self.d_model, 3 * self.num_mix_components)


    def encode(self, seq_obj : SequenceSeperate):
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

        if self.include_interarrival_time:
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

        return dtime_loss, num_predictions, None, None


    def loglike_loss_tf(self, batch, forward=True):

        # check if we are running validation or training
        if not forward:
            dtime_loss, num_predictions = self.loglike_loss_eval(batch)
            return dtime_loss, num_predictions

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]
 
        # Shift the input embeddings to the right
        shifted_embedding = torch.zeros_like(embedding)  # Initialize a zero tensor with the same shape as embedding
        shifted_embedding[:, 1:, :] = embedding[:, :-1, :]  # Shift embeddings to the right

        # Pack the shifted embeddings
        # shifted_embedding: [batch_size, seq_len, d_model]
        rnn_out, _ = self.layer_rnn(shifted_embedding)

        # filter out the src part
        tgt_rnn_out = rnn_out[:, self.src_seq_len-1:-1, :]

        # RNN outputs and loss computation
        # rnn_out: [batch_size, seq_len, d_model(*2 if bidirectional)]
        pred_dist = self.get_pred_distribution(tgt_rnn_out)  # Predict based on RNN output

        labels = seq_obj.tgt_dtime_seqs_transformed
        # Compute the log-likelihood loss
        assert labels.shape == pred_dist.mean.shape  # Ensure alignment
        dtime_ll = pred_dist.log_prob(labels)
        dtime_loss = -dtime_ll.sum()

        num_predictions = seq_obj.tgt_dtime_seqs_transformed.shape[0] * seq_obj.tgt_dtime_seqs_transformed.shape[1]

        return dtime_loss, num_predictions

    def get_rnn_tgt_input_step(self, rnn_out_step, seq_obj : SequenceSeperate, idx : int):
        """
        inputs
            rnn_out_step: [batch_size, 1, d_model] -> to get the delay distribution and the base input to the next step
            seq_obj: SequenceSeperate object
            idx (int): index of the target sequence we are predicting
        outputs
            input_step: [batch_size, 1, d_model]
        """
        pred_dist_step = self.get_pred_distribution(rnn_out_step)
        pred_dtime_step = pred_dist_step.mean
        pred_dtime_step_transformed = self.dtime_transform.inv(pred_dtime_step)
        enc_step = torch.zeros_like(rnn_out_step, device=self.device)
        # delay embedding (MUST EXIST)
        dtime_enc = self.dtime_emb_layer(pred_dtime_step_transformed.unsqueeze(-1))
        enc_step += dtime_enc

        slot, len, len_transformed, mcs, mretx, rfailed, num_rbs, time, dtime, \
            dtime_transformed, etype, interarrival_time, interarrival_time_transformed, \
            non_pad_mask, attention_mask = seq_obj.get_element_at_idx(self.src_seq_len + idx -1) # outputs all have [batch_size, 1]
        # self.src_seq_len + idx -1 because idx starts from 0 and -1 due to the shift in the target sequence

        # only linear ones need unsqueeze
        if self.include_interarrival_time:
            interarrival_time_enc = self.layer_interarrival_time_embedding(interarrival_time.unsqueeze(-1))
            enc_step += interarrival_time_enc

        if self.include_len:
            len_seqs = len.float().unsqueeze(-1) # applied inverse transform to len
            len_enc = self.layer_len_emb(len_seqs)
            enc_step += len_enc

        if self.include_time_embedding:
            time_enc = self.layer_time_embedding(time.unsqueeze(-1))
            enc_step += time_enc

        if self.include_slot:
            slot_seqs = slot.long()
            slot_enc = self.layer_slot_emb(slot_seqs)
            enc_step += slot_enc
        
        if self.include_mcs:
            if self.include_mcs_in_tgt:
                mcs_seqs = mcs.long()
                mcs_enc = self.layer_mcs_emb(mcs_seqs)
                enc_step += mcs_enc
            else:
                mcs_enc = self.layer_mcs_emb(self.mcs_pad_id * torch.ones_like(mcs, device=self.device, dtype=torch.long))
                enc_step += mcs_enc

        if self.include_mretx:
            if self.include_mretx_in_tgt:
                mretx_seqs = mretx.long()
                mretx_enc = self.layer_mretx_emb(mretx_seqs)
                enc_step += mretx_enc
            else:
                mretx_enc = self.layer_mretx_emb(self.mretx_pad_id * torch.ones_like(mretx, device=self.device, dtype=torch.long))
                enc_step += mretx_enc

        if self.include_rfailed:
            if self.include_rfailed_in_tgt:
                rfailed_seqs = rfailed.long()
                rfailed_enc = self.layer_rfailed_emb(rfailed_seqs)
                enc_step += rfailed_enc
            else:
                rfailed_enc = self.layer_rfailed_emb(self.rfailed_pad_id * torch.ones_like(rfailed, device=self.device, dtype=torch.long))
                enc_step += rfailed_enc

        input_step = enc_step
        return input_step

    def loglike_loss_eval(self, batch):

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embedding = torch.zeros_like(embedding)  # Initialize a zero tensor with the same shape as embedding
        shifted_embedding[:, 1:, :] = embedding[:, :-1, :]  # Shift embeddings to the right

        sh_src_embedding = shifted_embedding[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]

        # feed in the hisotry data
        rnn_out, prev_hidden = self.layer_rnn(sh_src_embedding)
        # [batch_size, his_len, d_model]
 
        # encode it to get the input for the next step
        input_step = self.get_rnn_input_step(rnn_out[:, -1:, :], seq_obj.tgt_interarrival_time_seqs_transformed[:,0].unsqueeze(-1))
        # input_step: [batch_size, 1, d_model]
        batch_size, seq_len, d_model = embedding.size()
        predictions = torch.zeros(batch_size, self.tgt_seq_len, d_model, device=self.device)
        predictions[:, 0, :] = rnn_out[:, -1, :]
        # We'll step one time at a time for the future steps.
        for i in range(self.tgt_seq_len-1):
            rnn_out_step, prev_hidden = self.layer_rnn(input_step, prev_hidden)
            # [batch_size, 1, d_model]
            predictions[:, i+1, :] = rnn_out_step[:, 0, :]
            input_step = self.get_rnn_input_step(rnn_out_step, seq_obj.tgt_interarrival_time_seqs_transformed[:,i+1].unsqueeze(-1))

        pred_dist = self.get_pred_distribution(predictions)
        labels = seq_obj.tgt_dtime_seqs_transformed

        # Apply prediction mask to filter out invalid positions
        assert labels.shape == pred_dist.mean.shape # [batch_size, seq_len]
        dtime_ll = pred_dist.log_prob(labels)
        dtime_loss = -dtime_ll.sum()

        num_predictions = labels.shape[0] * labels.shape[1]
        return dtime_loss, num_predictions


    def predict_mean_variance(self, batch):
        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embedding = torch.zeros_like(embedding)  # Initialize a zero tensor with the same shape as embedding
        shifted_embedding[:, 1:, :] = embedding[:, :-1, :]  # Shift embeddings to the right

        sh_src_embedding = shifted_embedding[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]

        # feed in the hisotry data
        rnn_out, prev_hidden = self.layer_rnn(sh_src_embedding)
        # [batch_size, 1, d_model]
 
        # encode it to get the input for the next step
        input_step = self.get_rnn_input_step(rnn_out[:, -1:, :], seq_obj.tgt_interarrival_time_seqs_transformed[:,0].unsqueeze(-1))
        # input_step: [batch_size, 1, d_model]
        batch_size, seq_len, d_model = embedding.size()
        predictions = torch.zeros(batch_size, self.tgt_seq_len, d_model, device=self.device)
        predictions[:, 0, :] = rnn_out[:, -1, :]
        # We'll step one time at a time for the future steps.
        for i in range(self.tgt_seq_len-1):
            rnn_out_step, prev_hidden = self.layer_rnn(input_step, prev_hidden)
            # [batch_size, 1, d_model]
            predictions[:, i+1, :] = rnn_out_step[:, 0, :]
            input_step = self.get_rnn_input_step(rnn_out_step, seq_obj.tgt_interarrival_time_seqs_transformed[:,i+1].unsqueeze(-1))

        pred_dist = self.get_pred_distribution(predictions)
        labels = seq_obj.tgt_dtime_seqs_transformed

        # Apply prediction mask to filter out invalid positions
        assert labels.shape == pred_dist.mean.shape # [batch_size, seq_len]

        pred_dtime = pred_dist.mean
        pred_dtime_var = pred_dist.variance

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), None, None