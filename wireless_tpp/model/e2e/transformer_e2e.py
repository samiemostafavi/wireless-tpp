import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FeedForwardBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_in)
        self.norm = nn.LayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # or ReLU()

    def forward(self, x):
        # pre-norm style
        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return x + out  # residual skip


class FeatureCombiner(nn.Module):
    def __init__(self, emb_dims, d_model):
        """
        emb_dims: list of dims for each feature's embedding
        d_model: final dimension
        """
        super().__init__()
        self.in_dim = sum(emb_dims)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, d_model)
        )

    def forward(self, *emb_list):
        # emb_list is list of Tensors [B, seq_len, d_featureDim]
        x = torch.cat(emb_list, dim=-1)  # [B, seq_len, sum(emb_dims)]
        return self.mlp(x)               # [B, seq_len, d_model]

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

    def get_target_seqs(self):
        return self.tgt_slot_seqs, self.tgt_len_seqs, self.tgt_len_seqs_transformed, self.tgt_mcs_seqs, self.tgt_mretx_seqs, self.tgt_rfailed_seqs, self.tgt_num_rbs_seqs, self.tgt_time_seqs, self.tgt_dtime_seqs, self.tgt_dtime_seqs_transformed, self.tgt_type_seqs, self.tgt_interarrival_time_seqs, self.tgt_interarrival_time_seqs_transformed, self.tgt_non_pad_mask, self.tgt_attention_mask

    def get_element_at_idx(self, idx):
        """
        input: idx (int) from 0 to self.tgt_seq_len+self.src_seq_len
        output: [batch_size, 1]
        """
        assert idx < self.tgt_seq_len+self.src_seq_len
        assert idx >= 0
        return self.slot_seqs[:,idx].unsqueeze(-1), self.len_seqs[:,idx].unsqueeze(-1), self.len_seqs_transformed[:,idx].unsqueeze(-1), self.mcs_seqs[:,idx].unsqueeze(-1), self.mretx_seqs[:,idx].unsqueeze(-1), self.rfailed_seqs[:,idx].unsqueeze(-1), self.num_rbs_seqs[:,idx].unsqueeze(-1), self.time_seqs[:,idx].unsqueeze(-1), self.dtime_seqs[:,idx].unsqueeze(-1), self.dtime_seqs_transformed[:,idx].unsqueeze(-1), self.type_seqs[:,idx].unsqueeze(-1), self.interarrival_time_seqs[:,idx].unsqueeze(-1), self.interarrival_time_seqs_transformed[:,idx].unsqueeze(-1), self.non_pad_mask[:,idx].unsqueeze(-1), self.attention_mask[:,idx].unsqueeze(-1)


class TransformerE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(TransformerE2E, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        self.mean_len = model_config.model_specs.get("mean_len", 0.0)
        self.std_len = model_config.model_specs.get("std_len", 1.0)
        self.mean_interarrival_time = model_config.model_specs.get("mean_interarrival_time", 0.0)
        self.std_interarrival_time = model_config.model_specs.get("std_interarrival_time", 1.0)
        logger.info(f"TransformerE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)
        logger.info(f"TransformerE2E loading mean and std of len: {self.mean_len}, {self.std_len}")
        self.len_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)
        logger.info(f"TransformerE2E loading mean and std of interarrival time: {self.mean_interarrival_time}, {self.std_interarrival_time}")
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
        self.last_layer_mlp = model_config.model_specs['last_layer_mlp']
        self.concat_features = model_config.model_specs['concat_features']

        # size of transformer tokens stays fixed
        self.n_encoder_heads = model_config.model_specs['encoder']['num_heads']
        self.n_encoder_layers = model_config.model_specs['encoder']['num_layers']
        self.encoder_use_residual = model_config.model_specs['encoder']['use_residual']
        logger.info(f"Encoder with {self.n_encoder_heads} heads, num_layers: {self.n_encoder_layers}, use residual: {self.encoder_use_residual}")
        if not self.last_layer_mlp:
            self.n_decoder_self_heads = model_config.model_specs['decoder']['num_self_heads']
            self.n_decoder_cross_heads = model_config.model_specs['decoder']['num_cross_heads']
            self.n_decoder_layers = model_config.model_specs['decoder']['num_layers']
            self.decoder_use_residual = model_config.model_specs['decoder']['use_residual']
            logger.info(f"Decoder with {self.n_decoder_self_heads} self heads, {self.n_decoder_cross_heads} cross heads, num_layers: {self.n_decoder_layers}, use residual: {self.decoder_use_residual}")

        # dtime embedding MUST EXIST
        self.dtime_emb_dim = self.d_model

        self.include_time_embedding = model_config.model_specs['embeddings']['include_time']
        self.time_emb_dim = self.d_model

        self.include_interarrival_time = model_config.model_specs['embeddings']['include_interarrival_time']
        self.interarrival_time_emb_dim = self.d_model

        self.include_len = model_config.model_specs['embeddings']['include_len']
        self.len_emb_dim = self.d_model
        
        self.include_slot = model_config.model_specs['embeddings']['include_slot']
        self.slot_emb_dim = self.d_model
        
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

        features_dims = []
        # Embedding layers defenitions
        # delay embedding layer
        self.dtime_emb_layer = nn.Linear(1, self.d_model)
        features_dims.append(self.d_model)

        # --- Add a standard positional encoder ---
        # Was very important to add this to the model
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, 
            max_len=1000, 
            dropout=self.dropout
        )
        self.pos_decoder = PositionalEncoding(
            d_model=self.d_model, 
            max_len=1000, 
            dropout=self.dropout
        )

        if self.include_time_embedding:
            self.layer_time_embedding = TimePositionalEncoding(
                self.time_emb_dim, device=self.device
            )
            features_dims.append(self.time_emb_dim)

        if self.include_interarrival_time:
            self.layer_interarrival_time_embedding = nn.Linear(
                1, 
                self.interarrival_time_emb_dim, 
                device=self.device
            )
            features_dims.append(self.interarrival_time_emb_dim)

        if self.include_slot:
            # slot number encoding
            self.layer_slot_emb = nn.Embedding(
                self.num_slots_types,
                self.slot_emb_dim,
                padding_idx=self.slots_pad_id,
                device=self.device
            )
            features_dims.append(self.slot_emb_dim)

        if self.include_mcs:
            # mcs encoding
            self.layer_mcs_emb = nn.Embedding(
                self.num_mcs_types,
                self.mcs_emb_dim,
                padding_idx=self.mcs_pad_id,
                device=self.device
            )
            features_dims.append(self.mcs_emb_dim)

        if self.include_mretx:
            # retransmissions encoding
            self.layer_mretx_emb = nn.Embedding(
                self.num_mretx_types,
                self.mretx_emb_dim,
                padding_idx=self.mretx_pad_id,
                device=self.device
            )
            features_dims.append(self.mretx_emb_dim)

        if self.include_rfailed:
            # failed attempt encoding
            self.layer_rfailed_emb = nn.Embedding(
                self.num_rfailed_types,
                self.rfailed_emb_dim, 
                padding_idx=self.rfailed_pad_id,
                device=self.device
            )
            features_dims.append(self.rfailed_emb_dim)

        if self.include_len:
            # length in bytes encoding (continuous)
            self.layer_len_emb = nn.Linear(
                1, 
                self.len_emb_dim, 
                device=self.device
            )
            features_dims.append(self.len_emb_dim)

        if self.concat_features:
            self.features_to_dmodel = FeatureCombiner(
                features_dims, 
                self.d_model
            )

        # Encoder layers
        # encoder MLP
        self.feed_forward_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4), # *4 is important, better than *2
            nn.GELU(), # THIS IS IMPORTANT
            nn.Dropout(p=self.dropout),
            nn.Linear(self.d_model * 4, self.d_model), # *4 is important, better than *2
            nn.Dropout(p=self.dropout), # SO IMPORTANT TO ADD DROPOUT HERE
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

        self.linear_layer_input_size = self.d_model
        if self.last_layer_mlp:
            # MLP layer
            #self.linear = nn.Sequential( # this is the old one when we didn't use MCS index and concat features
            #    nn.Linear(self.linear_layer_input_size, self.d_model * 4),  # Expand
            #    nn.ReLU(),
            #    nn.Dropout(0.1),
            #    nn.Linear(self.d_model * 4, self.d_model * 16),  # Middle layer (further expand)
            #    nn.ReLU(),
            #    nn.Dropout(0.1),
            #    nn.Linear(self.d_model * 16, self.tgt_seq_len * 3 * self.num_mix_components),  # Output projection
            #)
            self.linear = nn.Sequential( # this is the new one when we use MCS index and concat features, much better results
                nn.Linear(self.linear_layer_input_size, self.d_model * 4),
                nn.GELU(), # THIS IS IMPORTANT
                nn.Dropout(0.1),
                nn.Linear(self.d_model * 4, self.d_model * 8),
                FeedForwardBlock(self.d_model * 8, self.d_model * 16),
                FeedForwardBlock(self.d_model * 8, self.d_model * 16),
                nn.Linear(self.d_model * 8, self.tgt_seq_len * 3 * self.num_mix_components)
            )
        else:
            # Transformer decoder layers
            # decoder MLP
            self.feed_forward_decoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4), # *4 is important, better than *2
                nn.GELU(), # THIS IS IMPORTANT
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 4, self.d_model), # *4 is important, better than *2
                nn.Dropout(p=self.dropout), # SO IMPORTANT TO ADD DROPOUT HERE
            )
            # Transformer decoder layers (self.decoder_layers)
            self.decoder_layers = nn.ModuleList(
                [DecoderLayer(
                        d_model=self.d_model,
                        self_attn=MultiHeadAttention(self.n_decoder_self_heads, self.d_model, self.d_model,
                                                    dropout=self.dropout, output_linear=False),
                        cross_attn=MultiHeadAttention(self.n_decoder_cross_heads, self.d_model, self.d_model,
                                                    dropout=self.dropout, output_linear=False),
                        feed_forward=self.feed_forward_decoder,
                        use_residual=self.decoder_use_residual,
                        dropout=self.dropout
                ) for _ in range(self.n_decoder_layers)])
            # prediction linear layer
            self.linear = nn.Linear(self.linear_layer_input_size, 3 * self.num_mix_components)


    def get_embeddings(self, dtime, time, interarrival_time, slot, mcs, mretx, rfailed, len):
        """Get the embeddings for the input features. Does not apply positional encoding.

        Args:
            dtime (tensor): [batch_size, seq_len], delay times.
            time (tensor): [batch_size, seq_len], event times.
            interarrival_time (tensor): [batch_size, seq_len], interarrival times.
            slot (tensor): [batch_size, seq_len], slot indices.
            mcs (tensor): [batch_size, seq_len], mcs indices.
            mretx (tensor): [batch_size, seq_len], retransmission indices.
            rfailed (tensor): [batch_size, seq_len], failed attempt indices.
            len (tensor): [batch_size, seq_len], length in bytes.

        Returns:
            tensor: [batch_size, seq_len, d_model], embeddings.
        """
        batch_size, seq_len = dtime.size()
        # assert batch_size and seq_len are correct and equal among all
        assert time.size() == (batch_size, seq_len)
        assert interarrival_time.size() == (batch_size, seq_len)
        assert slot.size() == (batch_size, seq_len)
        assert mcs.size() == (batch_size, seq_len)
        assert mretx.size() == (batch_size, seq_len)
        assert rfailed.size() == (batch_size, seq_len)
        assert len.size() == (batch_size, seq_len)

        # only linear ones need unsqueeze
        # convert type_seqs to int type for embedding
        slot = slot.long()
        mcs = mcs.long()
        mretx = mretx.long()
        rfailed = rfailed.long()

        # output should be [batch_size, seq_len, hidden_size (d_model)]
        embeddings_sum = torch.zeros((batch_size,seq_len,self.d_model), device=self.device)
        features_emb = []

        dtime_enc = self.dtime_emb_layer(dtime.unsqueeze(-1))
        embeddings_sum += dtime_enc
        features_emb.append(dtime_enc)

        if self.include_time_embedding:
            time_enc = self.layer_time_embedding(time)
            embeddings_sum += time_enc
            features_emb.append(time_enc)

        if self.include_interarrival_time:
            interarrival_time_enc = self.layer_interarrival_time_embedding(interarrival_time.unsqueeze(-1))
            embeddings_sum += interarrival_time_enc
            features_emb.append(interarrival_time_enc)

        if self.include_slot: 
            slot_enc = self.layer_slot_emb(slot)
            embeddings_sum += slot_enc
            features_emb.append(slot_enc)
        
        if self.include_mcs:
            mcs_enc = self.layer_mcs_emb(mcs)
            embeddings_sum += mcs_enc
            features_emb.append(mcs_enc)

        if self.include_mretx:
            mretx_enc = self.layer_mretx_emb(mretx)
            embeddings_sum += mretx_enc
            features_emb.append(mretx_enc)

        if self.include_rfailed:
            rfailed_enc = self.layer_rfailed_emb(rfailed)
            embeddings_sum += rfailed_enc
            features_emb.append(rfailed_enc)

        if self.include_len:
            len_seqs = len.float().unsqueeze(-1)
            len_enc = self.layer_len_emb(len_seqs)
            embeddings_sum += len_enc
            features_emb.append(len_enc)

        if self.concat_features:
            combined = torch.cat(features_emb, dim=-1)
            final_embeddings = self.features_to_dmodel(combined)
        else:
            final_embeddings = embeddings_sum

        return final_embeddings

    def encode(self, seq_obj : SequenceSeperate):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, seq_len], attention masks.
        Returns:
            tensor: hidden states at event times.
        """

        embeddings = self.get_embeddings(
            seq_obj.dtime_seqs, seq_obj.time_seqs, seq_obj.interarrival_time_seqs, 
            seq_obj.slot_seqs, seq_obj.mcs_seqs, seq_obj.mretx_seqs, seq_obj.rfailed_seqs, seq_obj.len_seqs
        )
        embeddings = self.pos_encoder(embeddings)
        return embeddings

    def decode(self, dec_input_emb, enc_output, src_pad_mask, tgt_pad_mask):
        # dec_input_emb: [batch_size, tgt_seq_len, d_model]
        # enc_output: [batch_size, seq_len, d_model]
        # src_inp_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
        # tgt_inp_mask: [batch_size, tgt_seq_len] subsequent mask for decoder to prevent seeing future tokens

        # First, apply positional encoding to dec_input_emb
        dec_input_emb = self.pos_decoder(dec_input_emb)

        # fix mask_2d
        pad_positions_tgt = (tgt_pad_mask == 0)  # shape [B, T]
        pad_positions_src = (src_pad_mask == 0)  # shape [B, S]
        tgt_pad_2d = pad_positions_tgt.unsqueeze(2)      # [B, T, 1]
        src_pad_2d = pad_positions_src.unsqueeze(1)      # [B, 1, S]
        mask_2d = tgt_pad_2d | src_pad_2d           # [B, T, S], bool
        mask_2d = mask_2d.bool()  # [B, T, S], bool
        # shape [batch_size, tgt_seq_len, src_seq_len], True => "mask out"

        # Build the combined (pad + subsequent) mask for the decoder
        # input shape [batch_size, tgt_seq_len]
        # output shape [batch_size, tgt_seq_len, tgt_seq_len]
        tgt_mask = build_decoder_mask(tgt_pad_mask) 

        dec_output = dec_input_emb
        for idx, dec_layer in enumerate(self.decoder_layers):
            dec_output = dec_layer(
                dec_output,  # [batch_size, tgt_seq_len, d_model] is needed
                enc_output, # [batch_size, src_seq_len, d_model] is needed
                tgt_mask=tgt_mask,  # [batch_size, tgt_seq_len, tgt_seq_len] Mask for the target sequence (usually for preventing attention to future tokens)
                mask_2d=mask_2d # [batch_size, tgt_seq_len, src_seq_len] Mask for the cross attention (e.g., padding mask)
            )
        return dec_output



    def get_pred_distribution(self, dec_out, step=False) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
        if self.last_layer_mlp:
            rnn_out (tensor): [batch_size, d_model(*2 if bidirectional)]
        else:
            rnn_out (tensor): [batch_size, self.tgt_seq_len (1 if step), d_model(*2 if bidirectional)]

        Returns:  
            NormalMixtureDistribution: delay distribution with dim: [batch_size, self.tgt_seq_len (1 if step)]
        """
        if self.last_layer_mlp:
            assert dec_out.dim() == 2
            assert dec_out.size(1) == self.linear_layer_input_size
            # input: [batch_size, d_model(*2 if bidirectional)]
            # output: [batch_size, self.tgt_seq_len * 3 * num_mix_components]
            raw_params = self.linear(dec_out)

            # reshape to [batch_size, self.tgt_seq_len, 3 * num_mix_components]
            raw_params = raw_params.view(-1, self.tgt_seq_len, 3 * self.num_mix_components)
        else:
            assert dec_out.dim() == 3
            if step:
                assert dec_out.size(1) == 1
            else:
                assert dec_out.size(1) == self.tgt_seq_len
            assert dec_out.size(2) == self.linear_layer_input_size
            # input: [batch_size, self.tgt_seq_len, d_model(*2 if bidirectional)]
            # output: [batch_size, self.tgt_seq_len, 3 * num_mix_components]
            raw_params = self.linear(dec_out)

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

    def get_dec_input_tf(self, seq_obj : SequenceSeperate, idx : int):
        # Suppose self.tgt_dtime_seqs.shape = [batch_size, tgt_seq_len].
        batch_size, tgt_seq_len = seq_obj.tgt_dtime_seqs.size()

        # Create an all-PAD tensor
        dec_input = torch.full(
            (batch_size, tgt_seq_len, self.d_model), 
            fill_value=self.PAD_TOKEN,
            device=self.device
        )

        pad_mask = torch.full(
            (batch_size, tgt_seq_len), 
            fill_value=False,
            device=self.device
        )

        # Put the SOS token at position 0
        dec_input[:, 0, :] = torch.zeros((batch_size, self.d_model), device=self.device)
        pad_mask[:, 0] = True

        # Copy ground-truth dtimes up to idx-1 into positions [1..idx]
        # (Note: if idx=0, this does nothing.)
        if idx > 0:
            tmp = seq_obj.tgt_dtime_seqs[:, :idx]
            embeddings = self.get_embeddings(
                seq_obj.tgt_dtime_seqs[:, :idx], seq_obj.tgt_time_seqs[:, :idx], seq_obj.tgt_interarrival_time_seqs[:, :idx], seq_obj.tgt_slot_seqs[:, :idx], 
                seq_obj.tgt_mcs_seqs[:, :idx] if self.include_mcs_in_tgt else self.mcs_pad_id * torch.ones_like(tmp, device=self.device, dtype=torch.long), 
                seq_obj.mretx_seqs if self.include_mretx_in_tgt else self.mretx_pad_id * torch.ones_like(tmp, device=self.device, dtype=torch.long), 
                seq_obj.rfailed_seqs if self.include_rfailed_in_tgt else self.rfailed_pad_id * torch.ones_like(tmp, device=self.device, dtype=torch.long), 
                seq_obj.tgt_len_seqs[:, :idx]
            )
            dec_input[:, 1:idx+1] = embeddings
            pad_mask[:, 1:idx+1] = seq_obj.tgt_non_pad_mask[:, :idx]
        return dec_input, pad_mask.long()

    def append_dec_input(self, seq_obj : SequenceSeperate, idx : int, dec_out_step = None, prev_dec_input = None, prev_pad_mask = None):
        """
        inputs
            prev_dec_input: [batch_size, tgt_seq_len, d_model]
            prev_pad_mask: [batch_size, tgt_seq_len]
            dec_out_step: [batch_size, 1, d_model] -> to get the delay distribution and 
            the base input to the next step (not used in teacher forcing)
            seq_obj: SequenceSeperate object
            idx (int): index of the target sequence we are predicting
        outputs
            input_step: [batch_size, 1, d_model]
        """
        
        # Suppose self.tgt_dtime_seqs.shape = [batch_size, tgt_seq_len].
        batch_size, tgt_seq_len = seq_obj.tgt_dtime_seqs.size()

        if idx == 0:
            # Create an all-PAD tensor
            dec_input = torch.full(
                (batch_size, tgt_seq_len, self.d_model), 
                fill_value=self.PAD_TOKEN,
                device=self.device
            )

            pad_mask = torch.full(
                (batch_size, tgt_seq_len), 
                fill_value=False,
                device=self.device
            )

            # Put the SOS token at position 0
            dec_input[:, 0, :] = torch.zeros((batch_size, self.d_model), device=self.device)
            pad_mask[:, 0] = True

            return dec_input, pad_mask.long()
        
        elif idx > 0:
            pred_dist_step = self.get_pred_distribution(dec_out_step, step=True)
            pred_dtime_step_transformed = pred_dist_step.mean
            pred_dtime_step = self.dtime_transform.inv(pred_dtime_step_transformed)

            slot, len, len_transformed, mcs, mretx, rfailed, num_rbs, time, dtime, \
                dtime_transformed, etype, interarrival_time, interarrival_time_transformed, \
                non_pad_mask, attention_mask = seq_obj.get_element_at_idx(self.src_seq_len + idx -1) # outputs all have [batch_size, 1]
            # self.src_seq_len + idx -1 because idx starts from 0 and -1 due to the shift in the target sequence

            embeddings_step = self.get_embeddings(
                pred_dtime_step, time, interarrival_time, slot, 
                mcs if self.include_mcs_in_tgt else self.mcs_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
                mretx if self.include_mretx_in_tgt else self.mretx_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
                rfailed if self.include_rfailed_in_tgt else self.rfailed_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
                len
            )
            
            new_dec_input = prev_dec_input.clone()  # VERY IMPORTANT: clone to avoid in-place modifications
            new_pad_mask = prev_pad_mask.clone()

            new_dec_input[:, idx, :] = embeddings_step.squeeze(1)
            new_pad_mask[:, idx] = seq_obj.tgt_non_pad_mask[:, idx-1]

            return new_dec_input, new_pad_mask.long()

    def loglike_loss(self, batch, forward=True):

        # teacher forcing does not work for the last layer mlp
        if self.teacher_forcing:
            assert self.last_layer_mlp == False

        is_teacher_forcing_now = self.teacher_forcing
        if not forward:
            is_teacher_forcing_now = False

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.dtime_transform, self.len_transform, self.interarrival_time_transform)

        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding dims: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embedding = torch.zeros_like(embedding)  # Initialize a zero tensor with the same shape as embedding
        shifted_embedding[:, 1:, :] = embedding[:, :-1, :]  # Shift embeddings to the right
        sh_src_embedding = shifted_embedding[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]
        # sh_src_embedding dims: [batch_size, src_seq_len, d_model]

        # fix the mask
        src_non_pad_mask_float = seq_obj.src_non_pad_mask.float()  # Optional if it's not already in float
        src_attention_mask = src_non_pad_mask_float.unsqueeze(1) * src_non_pad_mask_float.unsqueeze(2)  # [batch_size, src_seq_len, src_seq_len, seq_len]
        src_attention_mask = src_attention_mask == 1

        # feed in the history data to the encoder
        # [batch_size, src_seq_len, hidden_size]
        enc_output = sh_src_embedding
        for idx, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(
                enc_output,
                mask=src_attention_mask
            )
        # enc_output dim: [batch_size, src_seq_len, d_model]
 
        if self.last_layer_mlp:
            # feed the last cell's output to MLP to predict the future
            pred_dist = self.get_pred_distribution(enc_output[:, -1, :], step=False)
            num_predictions = seq_obj.tgt_non_pad_mask.sum()
        else:
            # use decoder to predict the future
            # We'll store predictions for each time step
            all_preds = []
            num_predictions = 0
            # 3) Auto-regressive decoding
            #    for each idx in [0..(tgt_seq_len-1)], feed partial dec_input
            for idx in range(self.tgt_seq_len):
                # dec_input => [batch_size, tgt_seq_len], partial sequence up to idx-1
                # tgt_mask => [batch_size, tgt_seq_len], 1=real token, 0=pad token
                if is_teacher_forcing_now:
                    # does not use the output of the decoder
                    # just forces the labels to be the input of the decoder
                    # NOTE: should not be used for evaluation
                    dec_input, tgt_pad_mask = self.get_dec_input_tf(seq_obj, idx)
                else:
                    if idx == 0:
                        # produces the SOS token
                        dec_input, tgt_pad_mask = self.append_dec_input(
                            seq_obj=seq_obj, idx=0, dec_out_step=None, prev_dec_input=None, prev_pad_mask=None
                        )
                    else:
                        # takes dec_out_step, create a new embedding and append the result to the previous dec_input and pad_mask
                        dec_input, tgt_pad_mask = self.append_dec_input(
                            seq_obj=seq_obj, idx=idx, dec_out_step=dec_out_step.unsqueeze(1), prev_dec_input=dec_input, prev_pad_mask=tgt_pad_mask
                        )

                #print(seq_obj.src_non_pad_mask[0,:])
                #print(tgt_pad_mask[0,:])
                #print(dec_input[0,:,0])
                #print(enc_output[0,:,0])
                #input()
                
                # 4) Pass into the decoder
                # dec_input: [batch_size, tgt_seq_len]
                # enc_output: [batch_size, src_seq_len, d_model]
                # src_pad_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
                # tgt_pad_mask: [batch_size, tgt_seq_len] mask for decoder inputs (e.g. padding mask)
                # dec_out => [batch_size, tgt_seq_len, d_model]
                dec_out = self.decode(
                    dec_input_emb=dec_input,
                    enc_output=enc_output,
                    src_pad_mask=seq_obj.src_non_pad_mask.float(),
                    tgt_pad_mask=tgt_pad_mask
                )
                # we take the last position (i.e. dec_out[:, idx, :]) for prediction
                #dec_out_step = dec_out[:, idx, :] # shape [batch_size, d_model]
                dec_out_step = dec_out[:, -1, :]
                all_preds.append(dec_out_step)
                num_predictions += tgt_pad_mask[:, idx].sum()

            # and feed the results into a final linear to get distribution parameters.
            # 5) Convert all_preds => [batch_size, tgt_seq_len, d_model]
            all_preds = torch.stack(all_preds, dim=1)
            pred_dist = self.get_pred_distribution(all_preds, step=False)
            num_predictions = num_predictions.item()

        labels = seq_obj.tgt_dtime_seqs_transformed

        # Apply prediction mask to filter out invalid positions
        assert labels.shape == pred_dist.mean.shape # [batch_size, seq_len]
        dtime_ll = pred_dist.log_prob(labels)
        dtime_loss = -dtime_ll.sum()

        num_predictions = labels.shape[0] * labels.shape[1]
        return dtime_loss, num_predictions, None, None
    

    def predict_mean_variance(self, batch):

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding dims: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embedding = torch.zeros_like(embedding)  # Initialize a zero tensor with the same shape as embedding
        shifted_embedding[:, 1:, :] = embedding[:, :-1, :]  # Shift embeddings to the right
        sh_src_embedding = shifted_embedding[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]
        # sh_src_embedding dims: [batch_size, src_seq_len, d_model]

        # fix the mask
        src_non_pad_mask_float = seq_obj.src_non_pad_mask.float()  # Optional if it's not already in float
        src_attention_mask = src_non_pad_mask_float.unsqueeze(1) * src_non_pad_mask_float.unsqueeze(2)  # [batch_size, src_seq_len, src_seq_len, seq_len]
        src_attention_mask = src_attention_mask == 1

        # feed in the history data to the encoder
        # [batch_size, src_seq_len, hidden_size]
        enc_output = sh_src_embedding
        for idx, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(
                enc_output,
                mask=src_attention_mask
            )
        # enc_output dim: [batch_size, src_seq_len, d_model]
 
        if self.last_layer_mlp:
            # feed the last cell's output to MLP to predict the future
            pred_dist = self.get_pred_distribution(enc_output[:, -1, :], step=False)
            num_predictions = seq_obj.tgt_non_pad_mask.sum()
        else:
            # use decoder to predict the future
            # We'll store predictions for each time step
            all_preds = []
            num_predictions = 0
            # 3) Auto-regressive decoding
            #    for each idx in [0..(tgt_seq_len-1)], feed partial dec_input
            for idx in range(self.tgt_seq_len):
                # dec_input => [batch_size, tgt_seq_len], partial sequence up to idx-1
                # tgt_mask => [batch_size, tgt_seq_len], 1=real token, 0=pad token
                if idx == 0:
                    # produces the SOS token
                    dec_input, tgt_pad_mask = self.append_dec_input(
                        seq_obj=seq_obj, idx=0, dec_out_step=None, prev_dec_input=None, prev_pad_mask=None
                    )
                else:
                    # takes dec_out_step, create a new embedding and append the result to the previous dec_input and pad_mask
                    dec_input, tgt_pad_mask = self.append_dec_input(
                        seq_obj=seq_obj, idx=idx, dec_out_step=dec_out_step.unsqueeze(1), prev_dec_input=dec_input, prev_pad_mask=tgt_pad_mask
                    )

                #print(seq_obj.src_non_pad_mask[0,:])
                #print(tgt_pad_mask[0,:])
                #print(dec_input[0,:,0])
                
                # 4) Pass into the decoder
                # dec_input: [batch_size, tgt_seq_len]
                # enc_output: [batch_size, src_seq_len, d_model]
                # src_pad_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
                # tgt_pad_mask: [batch_size, tgt_seq_len] mask for decoder inputs (e.g. padding mask)
                # dec_out => [batch_size, tgt_seq_len, d_model]
                dec_out = self.decode(
                    dec_input_emb=dec_input,
                    enc_output=enc_output,
                    src_pad_mask=seq_obj.src_non_pad_mask.float(),
                    tgt_pad_mask=tgt_pad_mask
                )
                # we take the last position (i.e. dec_out[:, idx, :]) for prediction
                dec_out_step = dec_out[:, idx, :] # shape [batch_size, d_model]
                all_preds.append(dec_out_step)
                num_predictions += tgt_pad_mask[:, idx].sum()

            # and feed the results into a final linear to get distribution parameters.
            # 5) Convert all_preds => [batch_size, tgt_seq_len, d_model]
            all_preds = torch.stack(all_preds, dim=1)
            pred_dist = self.get_pred_distribution(all_preds, step=False)
            num_predictions = num_predictions.item()

        labels = seq_obj.tgt_dtime_seqs_transformed

        # Apply prediction mask to filter out invalid positions
        assert labels.shape == pred_dist.mean.shape # [batch_size, seq_len]

        pred_dtime = pred_dist.mean
        pred_dtime_var = pred_dist.variance

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), None, None

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