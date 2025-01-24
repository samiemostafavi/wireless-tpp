import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FeatureCombiner(nn.Module):
    def __init__(self, emb_dims, d_model, dropout_rate, device):
        """
        emb_dims: list of dims for each feature's embedding
        d_model: final dimension
        """
        super().__init__()
        self.in_dim = sum(emb_dims)
        #self.mlp = nn.Sequential(
        #    nn.Linear(self.in_dim, self.in_dim, device=device),
        #    nn.ReLU(),
        #    nn.Linear(self.in_dim, d_model, device=device)
        #)
        self.input_dropout = nn.Dropout(dropout_rate)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.in_dim, device=device),  # Expand
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * self.in_dim, self.in_dim, device=device),  # Compress
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_dim, d_model, device=device)           # Project to d_model
        )

    def forward(self, *emb_list):
        # emb_list is list of Tensors [B, seq_len, d_featureDim]
        x = torch.cat(emb_list, dim=-1)  # [B, seq_len, sum(emb_dims)]
        x = self.input_dropout(x)
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

        if self.src_seq_len == None or self.tgt_seq_len == None:
            return

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
        if self.src_seq_len == None or self.tgt_seq_len == None:
            return None
        return self.src_slot_seqs, self.src_len_seqs, self.src_len_seqs_transformed, self.src_mcs_seqs, self.src_mretx_seqs, self.src_rfailed_seqs, self.src_num_rbs_seqs, self.src_time_seqs, self.src_dtime_seqs, self.src_dtime_seqs_transformed, self.src_type_seqs, self.src_interarrival_time_seqs, self.src_interarrival_time_seqs_transformed, self.src_non_pad_mask, self.src_attention_mask

    def get_target_seqs(self):
        if self.src_seq_len == None or self.tgt_seq_len == None:
            return None
        return self.tgt_slot_seqs, self.tgt_len_seqs, self.tgt_len_seqs_transformed, self.tgt_mcs_seqs, self.tgt_mretx_seqs, self.tgt_rfailed_seqs, self.tgt_num_rbs_seqs, self.tgt_time_seqs, self.tgt_dtime_seqs, self.tgt_dtime_seqs_transformed, self.tgt_type_seqs, self.tgt_interarrival_time_seqs, self.tgt_interarrival_time_seqs_transformed, self.tgt_non_pad_mask, self.tgt_attention_mask

    def get_element_at_idx(self, idx):
        """
        input: idx (int) from 0 to self.tgt_seq_len+self.src_seq_len
        output: [batch_size, 1]
        """
        if self.src_seq_len == None or self.tgt_seq_len == None:
            return None
        assert idx < self.tgt_seq_len+self.src_seq_len
        assert idx >= 0
        return self.slot_seqs[:,idx].unsqueeze(-1), self.len_seqs[:,idx].unsqueeze(-1), self.len_seqs_transformed[:,idx].unsqueeze(-1), self.mcs_seqs[:,idx].unsqueeze(-1), self.mretx_seqs[:,idx].unsqueeze(-1), self.rfailed_seqs[:,idx].unsqueeze(-1), self.num_rbs_seqs[:,idx].unsqueeze(-1), self.time_seqs[:,idx].unsqueeze(-1), self.dtime_seqs[:,idx].unsqueeze(-1), self.dtime_seqs_transformed[:,idx].unsqueeze(-1), self.type_seqs[:,idx].unsqueeze(-1), self.interarrival_time_seqs[:,idx].unsqueeze(-1), self.interarrival_time_seqs_transformed[:,idx].unsqueeze(-1), self.non_pad_mask[:,idx].unsqueeze(-1), self.attention_mask[:,idx].unsqueeze(-1)


class DelayEmbedding(nn.Module):
    def __init__(self, d_model, device, model_config):
        super(DelayEmbedding, self).__init__()

        self.d_model = d_model
        self.device = device

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        self.mean_len = model_config.model_specs.get("mean_len", 0.0)
        self.std_len = model_config.model_specs.get("std_len", 1.0)
        self.mean_interarrival_time = model_config.model_specs.get("mean_interarrival_time", 0.0)
        self.std_interarrival_time = model_config.model_specs.get("std_interarrival_time", 1.0)
        logger.info(f"DelayEmbedding loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)
        logger.info(f"DelayEmbedding loading mean and std of len: {self.mean_len}, {self.std_len}")
        self.len_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)
        logger.info(f"DelayEmbedding loading mean and std of interarrival time: {self.mean_interarrival_time}, {self.std_interarrival_time}")
        self.interarrival_time_transform = D.AffineTransform(loc=self.mean_interarrival_time, scale=self.std_interarrival_time)

        # dtime embedding MUST EXIST for rnn and transformer
        self.include_dtime_embedding = model_config.model_specs['embeddings']['include_dtime']
        self.dtime_emb_dim = self.d_model

        self.concat_features = model_config.model_specs['concat_features']

        self.include_time_embedding = model_config.model_specs['embeddings']['include_time']
        self.time_emb_dim = self.d_model

        self.include_interarrival_time = model_config.model_specs['embeddings']['include_interarrival_time']
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

        # slots embedding
        self.num_slots_types = model_config.model_specs['types_and_paddings']['slot'][0]  
        self.slots_pad_id = model_config.model_specs['types_and_paddings']['slot'][1]
        # mcs embedding
        self.num_mcs_types = model_config.model_specs['types_and_paddings']['mcs'][0]  
        self.mcs_pad_id = model_config.model_specs['types_and_paddings']['mcs'][1]
        # retransmissions embedding
        self.num_mretx_types = model_config.model_specs['types_and_paddings']['mretx'][0]  
        self.mretx_pad_id = model_config.model_specs['types_and_paddings']['mretx'][1]
        # rlc failed embedding
        self.num_rfailed_types = model_config.model_specs['types_and_paddings']['rfailed'][0]  
        self.rfailed_pad_id = model_config.model_specs['types_and_paddings']['rfailed'][1]
        # rum rbs embedding
        self.num_rbs_types = model_config.model_specs['types_and_paddings']['num_rbs'][0] 
        self.rbs_pad_id = model_config.model_specs['types_and_paddings']['num_rbs'][1]

        features_dims = []
        # Embedding layers defenitions
        # delay embedding layer
        if self.include_dtime_embedding:
            self.dtime_emb_layer = nn.Linear(
                1, 
                self.dtime_emb_dim,
                device=self.device
            )
            features_dims.append(self.dtime_emb_dim)

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
                self.d_model,
                model_config.dropout_rate,
                self.device
            )

    def forward(self, dtime, time, interarrival_time, slot, mcs, mretx, rfailed, len):
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
        interarrival_time = interarrival_time.float()
        slot = slot.long()
        mcs = mcs.long()
        mretx = mretx.long()
        rfailed = rfailed.long()

        # output should be [batch_size, seq_len, hidden_size (d_model)]
        embeddings_sum = torch.zeros((batch_size,seq_len,self.d_model), device=self.device)
        features_emb = []

        if self.include_dtime_embedding:
            dtime_enc = self.dtime_emb_layer(dtime.float().unsqueeze(-1))
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
            len_enc = self.layer_len_emb(len.float().unsqueeze(-1))
            embeddings_sum += len_enc
            features_emb.append(len_enc)

        if self.concat_features:
            combined = torch.cat(features_emb, dim=-1)
            final_embeddings = self.features_to_dmodel(combined)
        else:
            final_embeddings = embeddings_sum

        return final_embeddings