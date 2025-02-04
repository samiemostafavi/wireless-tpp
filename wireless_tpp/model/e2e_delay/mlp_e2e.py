import numpy as np
import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.utils import RunnerPhase
from wireless_tpp.model.baselayer import FeedForwardBlock
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from .emb_utils import SequenceSeperate, DelayEmbedding
from .mdn_utils import MixtureDistribution


class MLPE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(MLPE2E, self).__init__(model_config)


        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate
        
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

        # only matters during prediction
        self.tgt_seq_len = model_config.model_specs['tgt_seq_len']
        self.src_seq_len = model_config.model_specs['src_seq_len']

        self.delay_embedding = DelayEmbedding(
            d_model=self.d_model,
            device = self.device,
            model_config = model_config
        )

        # MixtureDistribution
        self.mdn = MixtureDistribution(model_config, self.device)
        
        # MDN head layer 
        self.mdn_head = nn.Sequential( # this is the new one when we use MCS index and concat features, much better results
            nn.Linear(self.d_model, self.d_model * 4, device = self.device),
            FeedForwardBlock(self.d_model * 4, self.d_model * 8, device = self.device, dropout=self.dropout),
            FeedForwardBlock(self.d_model * 4, self.d_model * 8, device = self.device, dropout=self.dropout),
            nn.Linear(self.d_model * 4, self.mdn.num_params, device = self.device)
        )

    def embed(self, seq_obj : SequenceSeperate):
        """ Embedding layer.
        """
        embeddings = self.delay_embedding(
            seq_obj.src_dtime_seqs[:,-1:], # last history packet's delay (prev_dtime)
            seq_obj.tgt_time_seqs[:,:1], # first target packet
            seq_obj.tgt_interarrival_time_seqs[:,:1], # first target packet
            seq_obj.tgt_slot_seqs[:,:1], # first target packet
            seq_obj.tgt_mcs_seqs[:,:1], # first target packet
            seq_obj.tgt_mretx_seqs[:,:1], # first target packet
            seq_obj.tgt_rfailed_seqs[:,:1], # first target packet
            seq_obj.tgt_len_seqs[:,:1] # first target packet 
        )
        # embedding dims: [batch_size, 1, d_model]

        return embeddings

    def forward(self, seq_obj : SequenceSeperate):

        # apply embedding on the delay sequence
        embeds = self.embed(seq_obj)
        # embeds dim: [batch_size, 1, d_model]

        mdn_params = self.mdn_head(embeds)
        # mdn_params dim: [batch_size, self.mdn.num_params]

        num_predictions = seq_obj.tgt_non_pad_mask[:,:1].sum(axis=0)

        return mdn_params, num_predictions

    def loglike_loss(self, batch, phase):
        
        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.delay_embedding.dtime_transform, self.delay_embedding.len_transform, self.delay_embedding.interarrival_time_transform)

        mdn_params, num_predictions = self.forward(seq_obj)
        # mdn_params: [batch_size, 1, self.mdn.num_params]

        if phase == RunnerPhase.TRAIN or phase == RunnerPhase.VALIDATE:
            # we only consider the first element of the tgt sequence for ll loss

            labels = seq_obj.tgt_dtime_seqs_transformed[:, :1]
            # labels: [batch_size, 1]

            nll, nll_mask = self.mdn.negative_loglikelihood(mdn_params, labels, seq_obj.tgt_non_pad_mask[:, :1])
            num_predictions_nll = nll_mask.sum(axis=0)
            assert np.array_equal(num_predictions.cpu().numpy(), num_predictions_nll.cpu().numpy())

        else: # phase == RunnerPhase.EVALUATE or phase == RunnerPhase.PREDICT
            # here we calculate the negative log likelihood loss for all the labels in the tgt sequence length
            # the mdn_params we repeat the same for all the tgt sequence length

            # repeat the mdn_params for the tgt sequence length
            # input: [batch_size, 1, self.mdn.num_params]
            mdn_params = mdn_params.repeat(1, self.tgt_seq_len, 1)
            num_predictions = num_predictions.repeat(self.tgt_seq_len)
            # output: [batch_size, tgt_seq_len, self.mdn.num_params]

            labels = seq_obj.tgt_dtime_seqs_transformed
            # labels: [batch_size, tgt_seq_len]

            nll, nll_mask = self.mdn.negative_loglikelihood(mdn_params, labels, seq_obj.tgt_non_pad_mask)
            num_predictions_nll = nll_mask.sum(axis=0)
            assert np.array_equal(num_predictions.cpu().numpy(), num_predictions_nll.cpu().numpy())

        return nll, nll_mask

    def predict(self, batch):

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.delay_embedding.dtime_transform, self.delay_embedding.len_transform, self.delay_embedding.interarrival_time_transform)

        interarrival_time_src_seqs = seq_obj.tgt_interarrival_time_seqs_transformed
        len_src_seqs = seq_obj.tgt_len_seqs_transformed
        labels = seq_obj.tgt_dtime_seqs_transformed

        mdn_params, num_predictions = self.forward(seq_obj)
        # mdn_params: [batch_size, 1, self.mdn.num_params]

        # repeat the mdn_params for the tgt sequence length
        # input: [batch_size, 1, self.mdn.num_params]
        mdn_params = mdn_params.repeat(1, self.tgt_seq_len, 1)
        # output: [batch_size, tgt_seq_len, self.mdn.num_params]
        pred_mean, pred_var = self.mdn.mean_variance(mdn_params)

        pred_q99a = self.mdn.quantile(mdn_params,q=0.005)
        pred_q99b = self.mdn.quantile(mdn_params,q=0.995)

        pred_q9a = self.mdn.quantile(mdn_params,q=0.05)
        pred_q9b = self.mdn.quantile(mdn_params,q=0.95)

        pred_q7a = self.mdn.quantile(mdn_params,q=0.15)
        pred_q7b = self.mdn.quantile(mdn_params,q=0.85)

        pred_q5a = self.mdn.quantile(mdn_params,q=0.25)
        pred_q5b = self.mdn.quantile(mdn_params,q=0.75)

        pred_mask = seq_obj.tgt_non_pad_mask

        assert labels.shape == pred_mean.shape
        assert labels.shape == pred_var.shape

        return (pred_mean, pred_var, pred_q5a, pred_q5b, pred_q7a, pred_q7b, pred_q9a, pred_q9b, pred_q99a, pred_q99b), labels, (interarrival_time_src_seqs, len_src_seqs), pred_mask


