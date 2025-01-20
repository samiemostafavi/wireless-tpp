import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise



class Data():
    def __init__(self, batch, device, dtime_transform, len_transform, interarrival_time_transform):
        self.device = device
        self.dtime_transform = dtime_transform
        self.len_transform = len_transform
        self.interarrival_time_transform = interarrival_time_transform
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, interarrival_time_seqs_transformed, non_pad_mask, attention_mask = batch

        # only consider the last self.tgt_seq_len events in the target
        self.dtime_transformed = dtime_seqs_transformed[:,-1]
        self.dtime = self.dtime_transform.inv(self.dtime_transformed)
        self.interarrival_time_transformed = interarrival_time_seqs_transformed[:,-1]
        self.interarrival_time = self.interarrival_time_transform.inv(self.interarrival_time_transformed)
        self.len_transformed = len_seqs_transformed[:,-1]
        self.len = self.len_transform.inv(self.len_transformed)
        self.non_pad_mask = non_pad_mask[:,-1]
        self.slot = slot_seqs[:,-1]
        self.mcs = mcs_seqs[:,-1]
        self.mretx = mretx_seqs[:,-1]
        self.rfailed = rfailed_seqs[:,-1]
        self.time = time_seqs[:,-1]

    def get_all(self):
        return self.slot, self.len, self.len_transformed, self.mcs, self.mretx, self.rfailed, self.time, self.dtime, self.dtime_transformed, self.interarrival_time, self.interarrival_time_transformed, self.non_pad_mask
    

class MLPE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(MLPE2E, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        self.mean_len = model_config.model_specs.get("mean_len", 0.0)
        self.std_len = model_config.model_specs.get("std_len", 1.0)
        self.mean_interarrival_time = model_config.model_specs.get("mean_interarrival_time", 0.0)
        self.std_interarrival_time = model_config.model_specs.get("std_interarrival_time", 1.0)
        logger.info(f"MLPE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)
        logger.info(f"MLPE2E loading mean and std of len: {self.mean_len}, {self.std_len}")
        self.len_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)
        logger.info(f"MLPE2E loading mean and std of interarrival time: {self.mean_interarrival_time}, {self.std_interarrival_time}")
        self.interarrival_time_transform = D.AffineTransform(loc=self.mean_interarrival_time, scale=self.std_interarrival_time)

        # Noise regularization, only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.num_mix_components = model_config.model_specs['mdn']['num_mix_components_dtime']
        
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

        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate
        
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

        # Embedding layers defenitions
        # temporal encoding
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
        
        self.dtime_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.Linear(self.d_model * 2, 3 * self.num_mix_components)
        )

    def encode(self, data_obj : Data):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, seq_len], attention masks.
        Returns:
            tensor: hidden states at event times.
        """

        slot, len, len_transformed, mcs, mretx, rfailed, \
            time, dtime, dtime_transformed, interarrival_time, interarrival_time_transformed, non_pad_mask = data_obj.get_all()

        # only linear ones need unsqueeze
        # convert type_seqs to int type for embedding
        slot = slot.long()
        mcs = mcs.long()
        mretx = mretx.long()
        rfailed = rfailed.long()

        enc_output = torch.zeros(slot.size(0), self.d_model, device=self.device)

        # [batch_size, hidden_size (d_model)]
        # Optional feature encodings
        if self.include_time_embedding:
            time_enc = self.layer_time_embedding(time.unsqueeze(-1))
            enc_output += time_enc

        if self.include_interarrival_time_embedding:
            interarrival_time_enc = self.layer_interarrival_time_embedding(interarrival_time.unsqueeze(-1))
            enc_output += interarrival_time_enc

        if self.include_len:
            len_enc = self.layer_len_emb(len.unsqueeze(-1))
            enc_output += len_enc

        if self.include_slot: 
            slot_enc = self.layer_slot_emb(slot)
            enc_output += slot_enc
        
        if self.include_mcs:
            mcs_enc = self.layer_mcs_emb(mcs)
            enc_output += mcs_enc

        if self.include_mretx:
            mretx_enc = self.layer_mretx_emb(mretx)
            enc_output += mretx_enc

        if self.include_rfailed:
            rfailed_enc = self.layer_rfailed_emb(rfailed)
            enc_output += rfailed_enc

        return enc_output, non_pad_mask # [batch_size, d_model]

    def get_pred_distribution(self, raw_params) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            rnn_out (tensor): [batch_size, seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution.
        """
        # input: [batch_size, 3 * num_mix_components]

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


    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """

        data_obj = Data(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)

        # 1. embed the input events
        data_emb, src_pad_mask = self.encode(data_obj)
        # data_emb: [batch_size, d_model]
        # src_pad_mask: [batch_size]

        mlp_out = self.dtime_mlp(data_emb)
        # mlp_out: [batch_size, 3 * num_mix_components]
        pred_dist = self.get_pred_distribution(mlp_out)
        # result: [batch_size, ]
        
        # 6) Compute negative log-likelihood vs. the ground-truth times
        labels = data_obj.dtime_transformed  # [batch_size]
        assert labels.shape == pred_dist.mean.shape
        num_predictions = src_pad_mask.sum()

        dtime_ll = pred_dist.log_prob(labels) * src_pad_mask.long()
        dtime_loss = -dtime_ll.sum()
        
        return dtime_loss, num_predictions.item(), None, None


    def predict_mean_variance(self, batch, forward=False):

        data_obj = Data(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)

        # 1. embed the input events
        data_emb, src_pad_mask = self.encode(data_obj)
        # data_emb: [batch_size, d_model]
        # src_pad_mask: [batch_size]

        mlp_out = self.dtime_mlp(data_emb)
        # mlp_out: [batch_size, 3 * num_mix_components]
        pred_dist = self.get_pred_distribution(mlp_out)
        # result: [batch_size, ]
        
        # 6) Compute negative log-likelihood vs. the ground-truth times
        labels = data_obj.dtime_transformed  # [batch_size]
        num_predictions = src_pad_mask.sum()

        # 8) Stack the results: [batch_size, tgt_seq_len]
        pred_dtime = pred_dist.mean
        pred_dtime_var  = pred_dist.variance

        assert labels.shape == pred_dtime.shape
        assert labels.shape == pred_dtime_var.shape

        return (pred_dtime.unsqueeze(-1),pred_dtime_var.unsqueeze(-1)), (None,None), (labels.unsqueeze(-1), None), src_pad_mask.unsqueeze(-1), num_predictions.item()

