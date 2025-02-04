import numpy as np
import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding, FeedForwardBlock
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger
from wireless_tpp.utils import RunnerPhase

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .emb_utils import SequenceSeperate, DelayEmbedding
from .mdn_utils import MixtureDistribution

class RecurrentE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(RecurrentE2E, self).__init__(model_config)


        assert model_config.model_specs['rnn_type'] == 'lstm' or model_config.model_specs['rnn_type'] == 'gru' or model_config.model_specs['rnn_type'] == 'rnn'
        self.rnn_type = model_config.model_specs['rnn_type']
        self.num_layers = model_config.model_specs['num_layers']
        self.bidirectional = bool(model_config.model_specs['bidirectional'])
        logger.info(f"RNN type: {self.rnn_type}, num_layers: {self.num_layers}, bidirectional: {self.bidirectional}")

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate

        self.tgt_seq_len = model_config.model_specs['tgt_seq_len']
        self.src_seq_len = model_config.model_specs['src_seq_len']
        self.teacher_forcing = model_config.model_specs['teacher_forcing']
        self.last_layer_mlp = model_config.model_specs['last_layer_mlp']
        
        self.PAD_TOKEN = -1.0

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

        self.delay_embedding = DelayEmbedding(
            d_model=self.d_model,
            device = self.device,
            model_config = model_config
        )

        # MixtureDistribution
        self.mdn = MixtureDistribution(model_config, self.device)

        self.include_prev_dtime_in_tgt = model_config.model_specs['target']['include_prev_dtime']
        self.include_slot_in_tgt = model_config.model_specs['target']['include_slot']
        self.include_mcs_in_tgt = model_config.model_specs['target']['include_mcs']
        self.include_mretx_in_tgt = model_config.model_specs['target']['include_mretx']
        self.include_rfailed_in_tgt = model_config.model_specs['target']['include_rfailed']
        self.PAD_TOKEN = -1.0

        # RNN layer definition
        if self.rnn_type == 'lstm':
            self.layer_rnn = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                device=self.device,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.layer_rnn = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                device=self.device,
                batch_first=True
            )
        elif self.rnn_type == 'rnn':
            self.layer_rnn = nn.RNN(
                input_size=self.d_model,
                hidden_size=self.d_model,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                device=self.device,
                batch_first=True
            )

        if self.bidirectional:
            self.mdnhead_input_size = 2 * self.d_model
        else:
            self.mdnhead_input_size = self.d_model

        if self.last_layer_mlp:
            # MDN head layer 
            self.mdn_head = nn.Sequential( # this is the new one when we use MCS index and concat features, much better results
                nn.Linear(self.d_model, self.d_model * 4, device = self.device),
                nn.GELU(), # THIS IS IMPORTANT
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 4, self.d_model * 8, device = self.device),
                FeedForwardBlock(self.d_model * 8, self.d_model * 16, device = self.device, dropout=self.dropout),
                FeedForwardBlock(self.d_model * 8, self.d_model * 16, device = self.device, dropout=self.dropout),
                nn.Linear(self.d_model * 8, self.tgt_seq_len * self.mdn.num_params, device = self.device)
            )
        else:
            # prediction linear layer
            self.mdn_head = nn.Linear(self.mdnhead_input_size, self.mdn.num_params, device=self.device)


    def embed(self, seq_obj : SequenceSeperate):
        """Call the model

        Args:
            seq_obj
        Returns:
            tensor: hidden states at event times.
        """

        # first, prepare the prev_dtime_seqs, which is shifted to the right by 1 and padding on the first position
        # we pad the target part of the sequence as well
        dtime_seqs = seq_obj.dtime_seqs[:, -self.src_seq_len-self.tgt_seq_len:]
        prev_dtime_seqs = torch.zeros_like(dtime_seqs)
        prev_dtime_seqs[:, 1:] = dtime_seqs[:, :-1]
        prev_dtime_seqs[:, 0] = self.PAD_TOKEN
        prev_dtime_seqs[:, self.src_seq_len:] = self.PAD_TOKEN

        # fix interarrival_time_seqs and time_seqs
        interarrival_time_seqs = seq_obj.interarrival_time_seqs[:, -self.src_seq_len-self.tgt_seq_len:]
        time_seqs = seq_obj.time_seqs[:, -self.src_seq_len-self.tgt_seq_len:] # it is never used
        len_seqs = seq_obj.len_seqs[:, -self.src_seq_len-self.tgt_seq_len:]

        # now that we have all sequences ready, we should replace the tgt part of some of the sequences with the paddings
        slot_seqs = torch.cat(
            [
                seq_obj.src_slot_seqs, 
                seq_obj.tgt_slot_seqs if self.include_slot_in_tgt else self.slots_pad_id * torch.ones_like(seq_obj.tgt_dtime_seqs, device=self.device, dtype=torch.long)
            ],
            dim=1
        )

        mcs_seqs = torch.cat(
            [
                seq_obj.src_mcs_seqs, 
                seq_obj.tgt_mcs_seqs if self.include_mcs_in_tgt else self.mcs_pad_id * torch.ones_like(seq_obj.tgt_dtime_seqs, device=self.device, dtype=torch.long)
            ],
            dim=1
        )
        mretx_seqs = torch.cat(
            [
                seq_obj.src_mretx_seqs, 
                seq_obj.tgt_mretx_seqs if self.include_mretx_in_tgt else self.mretx_pad_id * torch.ones_like(seq_obj.tgt_dtime_seqs, device=self.device, dtype=torch.long)
            ],
            dim=1
        )
        rfailed_seqs = torch.cat(
            [
                seq_obj.src_rfailed_seqs, 
                seq_obj.tgt_rfailed_seqs if self.include_rfailed_in_tgt else self.rfailed_pad_id * torch.ones_like(seq_obj.tgt_dtime_seqs, device=self.device, dtype=torch.long)
            ],
            dim=1
        )

        # apply embedding on the whole sequences (seq_len = src + tgt)
        embeddings = self.delay_embedding(
            prev_dtime_seqs,  # prev_dtime
            time_seqs, 
            interarrival_time_seqs, 
            slot_seqs, 
            mcs_seqs, 
            mretx_seqs, 
            rfailed_seqs, 
            len_seqs
        )
        # embedding dims: [batch_size, seq_len, d_model]

        # return the embeddings
        return embeddings

    def forward(self, seq_obj : SequenceSeperate, phase=None):

        if phase == RunnerPhase.TRAIN:
            forward = True
        else:
            forward = False

        # teacher forcing does not work for the last layer mlp
        if self.teacher_forcing:
            assert self.last_layer_mlp == False

        is_teacher_forcing_now = self.teacher_forcing
        if not forward:
            is_teacher_forcing_now = False

        if is_teacher_forcing_now:

            # apply embedding on the delay sequence
            embeddings = self.embed(seq_obj)
            # embeddings dim: [batch_size, seq_len = src_len + tgt_len, d_model]

            # embeddings: [batch_size, seq_len, d_model]
            rnn_out, _ = self.layer_rnn(embeddings)

            # filter out the src part
            # tgt_rnn_out: [batch_size, tgt_seq_len, d_model]
            tgt_rnn_out = rnn_out[:, self.src_seq_len:, :]
            mdn_params = self.mdn_head(tgt_rnn_out)
            # raw_params: [batch_size, tgt_seq_len, self.mdn.num_params]


        # apply embedding on the delay sequence
        embeddings = self.embed(seq_obj)
        # embeddings dim: [batch_size, seq_len = src_len + tgt_len, d_model]

        if self.last_layer_mlp:                
            # extract the src embedding
            embeddings_src = embeddings[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]

            # feed in the src data
            rnn_out, prev_hidden = self.layer_rnn(embeddings_src)
            # [batch_size, his_len, d_model]
    
            # feed the last cell's output to MLP
            mdn_params = self.mdn_head(rnn_out[:, -1:, :])

            # output is [batch_size, 1, self.tgt_seq_len*self.mdn.num_params]
            # convert it to [batch_size, self.tgt_seq_len, self.mdn.num_params]
            mdn_params = mdn_params.view(-1, self.tgt_seq_len, self.mdn.num_params)
        else:
            if self.include_prev_dtime_in_tgt:
                # extract the src embedding
                embeddings_src = embeddings[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]

                # feed in the src data
                rnn_out, prev_hidden = self.layer_rnn(embeddings_src)
                # [batch_size, his_len, d_model]

                # encode it to get the input for the next step
                input_step = self.get_rnn_tgt_input_step(rnn_out[:, -1:, :], seq_obj, 0)
                # input_step: [batch_size, 1, d_model]
                all_preds = [ rnn_out[:, -1, :] ]
                # We'll step one time at a time for the future steps.
                for i in range(self.tgt_seq_len-1):
                    rnn_out_step, prev_hidden = self.layer_rnn(input_step, prev_hidden)
                    # rnn_out_step dim: [batch_size, 1, d_model]
                    all_preds.append(rnn_out_step[:, 0, :])
                    input_step = self.get_rnn_tgt_input_step(rnn_out_step, seq_obj, i+1)
                all_preds = torch.stack(all_preds, dim=1)
                mdn_params = self.mdn_head(all_preds)
            else:

                # feed in the src data
                rnn_out, prev_hidden = self.layer_rnn(embeddings)
                # [batch_size, seq_len, d_model]

                # filter out the src part
                # tgt_rnn_out: [batch_size, tgt_len, d_model]
                tgt_rnn_out = rnn_out[:, self.src_seq_len:, :]
                mdn_params = self.mdn_head(tgt_rnn_out)
                # raw_params: [batch_size, tgt_seq_len, self.mdn.num_params]

        num_predictions = seq_obj.tgt_non_pad_mask.float().sum(axis=0)
        return mdn_params, num_predictions

    
    def get_rnn_tgt_input_step(self, rnn_out_step, seq_obj : SequenceSeperate, idx : int):
        """
        inputs
            rnn_out_step: [batch_size, 1, d_model] -> to get the delay distribution and the base input to the next step
            seq_obj: SequenceSeperate object
            idx (int): index of the target sequence we are predicting
        outputs
            input_step: [batch_size, 1, d_model]
        """
        raw_params = self.mdn_head(rnn_out_step)
        pred_dist_step = self.mdn(raw_params)
        pred_dtime_step_transformed = pred_dist_step.mean
        pred_dtime_step = self.delay_embedding.dtime_transform.inv(pred_dtime_step_transformed)

        slot, len, len_transformed, mcs, mretx, rfailed, num_rbs, time, dtime, \
            dtime_transformed, etype, interarrival_time, interarrival_time_transformed, \
            non_pad_mask, attention_mask = seq_obj.get_element_at_idx(self.src_seq_len + idx) # outputs all have [batch_size, 1]

        embeddings_step = self.delay_embedding(
            pred_dtime_step, # this will be prev_dtime
            time, 
            interarrival_time, 
            slot if self.include_slot_in_tgt else self.slots_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long),
            mcs if self.include_mcs_in_tgt else self.mcs_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
            mretx if self.include_mretx_in_tgt else self.mretx_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
            rfailed if self.include_rfailed_in_tgt else self.rfailed_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
            len
        )

        input_step = embeddings_step
        return input_step

    def loglike_loss(self, batch, phase):

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.delay_embedding.dtime_transform, self.delay_embedding.len_transform, self.delay_embedding.interarrival_time_transform)

        mdn_params, num_predictions = self.forward(seq_obj, phase)
        # mdn_params: [batch_size, tgt_seq_len, self.mdn.num_params]

        labels = seq_obj.tgt_dtime_seqs_transformed
        # labels: [batch_size, tgt_seq_len]

        nll, nll_mask = self.mdn.negative_loglikelihood(mdn_params, labels, seq_obj.tgt_non_pad_mask)
        num_predictions_nll = nll_mask.sum(axis=0)
        assert np.array_equal(num_predictions.cpu().numpy(), num_predictions_nll.cpu().numpy())

        return nll, nll_mask


    def predict(self, batch):

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.delay_embedding.dtime_transform, self.delay_embedding.len_transform, self.delay_embedding.interarrival_time_transform)
        label = seq_obj.tgt_dtime_seqs_transformed

        interarrival_time_src_seqs = seq_obj.src_interarrival_time_seqs_transformed
        len_src_seqs = seq_obj.src_len_seqs_transformed

        mdn_params, num_predictions = self.forward(seq_obj)
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

        assert label.shape == pred_mean.shape
        assert label.shape == pred_var.shape

        return (pred_mean, pred_var, pred_q5a, pred_q5b, pred_q7a, pred_q7b, pred_q9a, pred_q9b, pred_q99a, pred_q99b), label, (interarrival_time_src_seqs, len_src_seqs), pred_mask