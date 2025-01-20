import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise

class Sequence():
    def __init__(self, batch, device, dtime_transform, len_transform):
        self.PAD_TOKEN = -1.0
        self.SOS_TOKEN = 0.0

        self.device = device
        self.dtime_transform = dtime_transform
        self.len_transform = len_transform
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.tgt_seq_len events in the target
        self.dtime_seqs_transformed = dtime_seqs_transformed
        self.dtime_seqs = self.dtime_transform.inv(self.dtime_seqs_transformed)
        self.len_seqs_transformed = len_seqs_transformed
        self.len_seqs = self.len_transform.inv(self.len_seqs_transformed)
        self.batch_non_pad_mask = batch_non_pad_mask
        self.slot_seqs = slot_seqs
        self.mcs_seqs = mcs_seqs
        self.mretx_seqs = mretx_seqs
        self.rfailed_seqs = rfailed_seqs
        self.num_rbs_seqs = num_rbs_seqs
        self.time_seqs = time_seqs
        self.type_seqs = type_seqs
        self.attention_mask = attention_mask

    def get_all(self):
        return self.slot_seqs, self.len_seqs_transformed, self.mcs_seqs, self.mretx_seqs, self.rfailed_seqs, self.num_rbs_seqs, self.time_seqs, self.dtime_seqs_transformed, self.type_seqs, self.batch_non_pad_mask, self.attention_mask
    

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
        logger.info(f"FullTransformerE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)
        logger.info(f"FullTransformerE2E loading mean and std of len: {self.mean_len}, {self.std_len}")
        self.len_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)

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

        self.type_emb_dim = model_config.model_specs['embeddings']['type_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_len = model_config.model_specs['history']['include_len']
        self.len_emb_dim = model_config.model_specs['embeddings']['len_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_slot = model_config.model_specs['history']['include_slot']
        self.slot_emb_dim = model_config.model_specs['embeddings']['slot_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_mcs = model_config.model_specs['history']['include_mcs']
        self.mcs_emb_dim = model_config.model_specs['embeddings']['mcs_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_mretx = model_config.model_specs['history']['include_mretx']
        self.mretx_emb_dim = model_config.model_specs['embeddings']['mretx_emb_dim'] if self.concat_embeddings else self.d_model

        self.include_rfailed = model_config.model_specs['history']['include_rfailed']
        self.rfailed_emb_dim = model_config.model_specs['embeddings']['rfailed_emb_dim'] if self.concat_embeddings else self.d_model

        self.include_num_rbs = model_config.model_specs['history']['include_num_rbs']
        self.num_rbs_emb_dim = model_config.model_specs['embeddings']['num_rbs_emb_dim'] if self.concat_embeddings else self.d_model

        if self.concat_embeddings:
            # size of time embedding is self.d_model minues the total size of the other embeddings
            self.time_emb_size = self.d_model - (
                int(self.include_slot)*self.slot_emb_dim + \
                int(self.include_mcs)*self.mcs_emb_dim + \
                int(self.include_mretx)*self.mretx_emb_dim + \
                int(self.include_rfailed)*self.rfailed_emb_dim + \
                int(self.include_num_rbs)*self.num_rbs_emb_dim + \
                int(self.include_len)*self.len_emb_dim + \
                self.type_emb_dim
            )
        else:
            self.time_emb_size = self.d_model
        
        self.n_encoder_heads = model_config.model_specs['encoder']['num_heads']
        self.n_encoder_layers = model_config.model_specs['encoder']['num_layers']
        self.encoder_use_residual = model_config.model_specs['encoder']['use_residual']

        self.seq_len = model_config.model_specs['tgt_seq_len']
        self.dropout = model_config.dropout_rate
        
        self.PAD_TOKEN = -1.0
        self.SOS_TOKEN = 0.0
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

        self.num_event_types_pad = 2 # should be 4
        self.pad_token_id = 1
        # Embedding layers defenitions
        # temporal encoding
        self.layer_temporal_encoding = TimePositionalEncoding(
            self.time_emb_size, device=self.device
        )
        # type embedding
        self.layer_type_emb = nn.Embedding(
            self.num_event_types_pad,
            self.type_emb_dim,
            padding_idx=self.pad_token_id,
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
        if self.include_num_rbs:
            # number of rbs encoding
            self.layer_num_rbs_emb = nn.Embedding(
                self.num_rbs_types,
                self.num_rbs_emb_dim,
                padding_idx=self.rbs_pad_id,
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
        
        # prediction linear layers
        self.dtime_linear = nn.Linear(self.d_model, 3 * self.num_mix_components_dtime, device=self.device)
        #self.dtime_linear = nn.Sequential(
        #    nn.Linear(self.d_model, self.d_model * 2),
        #    nn.ReLU(),
        #    nn.Linear(self.d_model * 2, 3 * self.num_mix_components_dtime)
        #)

        # prediction linear layers
        #self.dtime_linears = [ 
        #    nn.Linear(self.d_model, 3 * self.num_mix_components_dtime, device=self.device) for _ in range(self.seq_len)
        #]

    def encode(self, seq_obj : Sequence):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, seq_len], attention masks.
        Returns:
            tensor: hidden states at event times.
        """

        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, \
            num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, \
            non_pad_mask, causal_mask = seq_obj.get_all()

        # fix the mask
        non_pad_mask_float = non_pad_mask.float()  # Optional if it's not already in float
        attention_mask = non_pad_mask_float.unsqueeze(1) * non_pad_mask_float.unsqueeze(2)  # [batch_size, seq_len, seq_len]
        attention_mask = attention_mask == 1
        #combined_mask = (~causal_mask) & attention_mask

        # only linear ones need unsqueeze
        # convert type_seqs to int type for embedding
        type_seqs = type_seqs.long()
        slot_seqs = slot_seqs.long()
        mcs_seqs = mcs_seqs.long()
        mretx_seqs = mretx_seqs.long()
        rfailed_seqs = rfailed_seqs.long()
        num_rbs_seqs = num_rbs_seqs.long()

        len_seqs = seq_obj.len_seqs # applied inverse transform to len
        len_seqs = len_seqs.float().unsqueeze(-1)

        # [batch_size, seq_len, hidden_size (d_model)]
        # Temporal and type encoding
        time_enc = self.layer_temporal_encoding(time_seqs)
        type_enc = self.layer_type_emb(type_seqs) # it is either packet arrival, first segment or segments later

        # 2) Build a list to concatenate later (maybe)
        emb_list = [time_enc, type_enc]

        # Optional feature encodings
        if self.include_len:
            len_enc = self.layer_len_emb(len_seqs)
            emb_list.append(len_enc)
        else:
            len_enc = 0

        if self.include_slot: 
            slot_enc = self.layer_slot_emb(slot_seqs)
            emb_list.append(slot_enc)
        else:
            slot_enc = 0
        
        if self.include_mcs:
            mcs_enc = self.layer_mcs_emb(mcs_seqs)
            emb_list.append(mcs_enc)
        else:
            mcs_enc = 0

        if self.include_mretx:
            mretx_enc = self.layer_mretx_emb(mretx_seqs)
            emb_list.append(mretx_enc)
        else:
            mretx_enc = 0

        if self.include_rfailed:
            rfailed_enc = self.layer_rfailed_emb(rfailed_seqs)
            emb_list.append(rfailed_enc)
        else:
            rfailed_enc = 0

        if self.include_num_rbs:
            num_rbs_enc = self.layer_num_rbs_emb(num_rbs_seqs)
            emb_list.append(num_rbs_enc)
        else:
            num_rbs_enc = 0

        if self.concat_embeddings:
            # 3) Concatenate along the last dimension
            # shape -> [B, S, sum_of_emb_dims]
            enc_output = torch.cat(emb_list, dim=-1)
            # [batch_size, seq_len, hidden_size]
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(
                    enc_output,
                    mask=attention_mask
                )
        else:
            #enc_output = type_enc + slot_enc + mcs_enc + mretx_enc + rfailed_enc + len_enc + num_rbs_enc
            enc_output = type_enc
            if self.include_len:
                enc_output += len_enc
            if self.include_slot:
                enc_output += slot_enc
            if self.include_mcs:
                enc_output += mcs_enc
            if self.include_mretx:
                enc_output += mretx_enc
            if self.include_rfailed:
                enc_output += rfailed_enc
            if self.include_num_rbs:
                enc_output += num_rbs_enc

            # [batch_size, seq_len, hidden_size]
            for enc_layer in self.encoder_layers:
                enc_output += time_enc
                enc_output = enc_layer(
                    enc_output,
                    mask=attention_mask
                )

        # encoder_mask: shape [batch_size, seq_len, seq_len]
        # 1 => masked, 0 => not masked

        # We can say: "A source token is considered padded if the entire row is masked."
        # or if the diagonal is masked. It depends on how you built it.

        src_pad_mask_1d = non_pad_mask_float   # shape [batch_size, seq_len]
        return enc_output, src_pad_mask_1d
    

    def get_pred_distribution(self, enc_out, src_pad_mask) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            enc_out (tensor): [batch_size, seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution.
        """
        # input: [batch_size, seq_len, hidden_size]
        # output: [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.dtime_linear(enc_out)

        # input: [batch_size, seq_len, hidden_size]
        #raw_params = []
        #for i in range(self.seq_len):
        #    raw_params.append(self.dtime_linears[i](enc_out[:,i,:]))
        #raw_params = torch.stack(raw_params, dim=1)
        # output: [batch_size, seq_len, 3 * num_mix_components]

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

        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform)
        self.seq_len = seq_obj.dtime_seqs.size(1)

        # 1. encode the history
        # enc_out: [batch_size, seq_len, hidden_size]
        # src_mask: [batch_size, seq_len]
        enc_out, src_pad_mask = self.encode(seq_obj)
        # enc_out: [batch_size, seq_len, hidden_size]
        #enc_out = enc_out * src_pad_mask.unsqueeze(-1)

        pred_dist = self.get_pred_distribution(enc_out, src_pad_mask)
        # result: [batch_size, seq_len]
        
        # 6) Compute negative log-likelihood vs. the ground-truth times
        labels = seq_obj.dtime_seqs_transformed  # [batch_size, seq_len]
        assert labels.shape == pred_dist.mean.shape
        num_predictions = seq_obj.batch_non_pad_mask.sum()

        dtime_ll = pred_dist.log_prob(labels)  * seq_obj.batch_non_pad_mask.float()
        dtime_loss = -dtime_ll.sum()
        
        return dtime_loss, num_predictions.item(), None, None


    def predict_mean_variance(self, batch, forward=False):
        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform)

        # 1. encode the history
        # enc_out: [batch_size, seq_len, hidden_size]
        # src_mask: [batch_size, seq_len]
        enc_out, src_pad_mask = self.encode(seq_obj)
        # enc_out: [batch_size, seq_len, hidden_size]

        pred_dist = self.get_pred_distribution(enc_out, src_pad_mask)
        # result: [batch_size, seq_len]

        # 8) Stack the results: [batch_size, tgt_seq_len]
        pred_dtime = pred_dist.mean
        pred_dtime_var  = pred_dist.variance

        labels = seq_obj.dtime_seqs_transformed  # [batch_size, tgt_seq_len]
        assert labels.shape == pred_dtime.shape
        assert labels.shape == pred_dtime_var.shape

        num_predictions = seq_obj.batch_non_pad_mask.sum()

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), seq_obj.batch_non_pad_mask, num_predictions.item()
