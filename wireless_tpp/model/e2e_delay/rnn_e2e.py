import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding, FeedForwardBlock
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

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
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, seq_len], attention masks.
        Returns:
            tensor: hidden states at event times.
        """

        embeddings = self.delay_embedding(
            seq_obj.dtime_seqs, seq_obj.time_seqs, seq_obj.interarrival_time_seqs, 
            seq_obj.slot_seqs, seq_obj.mcs_seqs, seq_obj.mretx_seqs, seq_obj.rfailed_seqs, seq_obj.len_seqs
        )
        # embedding dims: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embeds = torch.zeros_like(embeddings)  # Initialize a zero tensor with the same shape as embedding
        shifted_embeds[:, 1:, :] = embeddings[:, :-1, :]  # Shift embeddings to the right
        # sh_src_embedding dims: [batch_size, seq_len, d_model]

        return shifted_embeds

    def forward(self, seq_obj : SequenceSeperate, forward=True):

        # teacher forcing does not work for the last layer mlp
        if self.teacher_forcing:
            assert self.last_layer_mlp == False

        is_teacher_forcing_now = self.teacher_forcing
        if not forward:
            is_teacher_forcing_now = False

        # apply embedding on the delay sequence
        shifted_embeds = self.embed(seq_obj)
        # shifted_embeds dim: [batch_size, seq_len, d_model]

        if is_teacher_forcing_now:
            # shifted_embedding: [batch_size, seq_len, d_model]
            rnn_out, _ = self.layer_rnn(shifted_embeds)

            # filter out the src part
            # tgt_rnn_out: [batch_size, tgt_seq_len, d_model]
            tgt_rnn_out = rnn_out[:, self.src_seq_len:, :]
            mdn_params = self.mdn_head(tgt_rnn_out)
            # raw_params: [batch_size, tgt_seq_len, self.mdn.num_params]
        else:
            # extract the src embedding
            sh_src_embedding = shifted_embeds[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]
            # feed in the src data
            rnn_out, prev_hidden = self.layer_rnn(sh_src_embedding)
            # [batch_size, his_len, d_model]
    
            if self.last_layer_mlp:
                # feed the last cell's output to MLP
                mdn_params = self.mdn_head(rnn_out[:, -1:, :])
                # output is [batch_size, 1, self.tgt_seq_len*self.mdn.num_params]
                # convert it to [batch_size, self.tgt_seq_len, self.mdn.num_params]
                mdn_params = mdn_params.view(-1, self.tgt_seq_len, self.mdn.num_params)
            else:
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

        num_predictions = seq_obj.tgt_non_pad_mask.float().sum()
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
            non_pad_mask, attention_mask = seq_obj.get_element_at_idx(self.src_seq_len + idx -1) # outputs all have [batch_size, 1]
        # self.src_seq_len + idx -1 because idx starts from 0 and -1 due to the shift in the target sequence

        embeddings_step = self.delay_embedding(
            pred_dtime_step, 
            time, 
            interarrival_time, 
            slot, 
            mcs if self.include_mcs_in_tgt else self.mcs_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
            mretx if self.include_mretx_in_tgt else self.mretx_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
            rfailed if self.include_rfailed_in_tgt else self.rfailed_pad_id * torch.ones_like(pred_dtime_step, device=self.device, dtype=torch.long), 
            len
        )

        input_step = embeddings_step
        return input_step

    def loglike_loss(self, batch, forward=True):

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.delay_embedding.dtime_transform, self.delay_embedding.len_transform, self.delay_embedding.interarrival_time_transform)

        mdn_params, num_predictions = self.forward(seq_obj, forward=forward)
        # mdn_params: [batch_size, tgt_seq_len, self.mdn.num_params]

        labels = seq_obj.tgt_dtime_seqs_transformed
        # labels: [batch_size, tgt_seq_len]

        nll, num_predictions_nll = self.mdn.negative_loglikelihood(mdn_params, labels, seq_obj.tgt_non_pad_mask)
        assert num_predictions.item() == num_predictions_nll.item()

        return nll, num_predictions, None, None


    def predict_mean_variance(self, batch):

        seq_obj = SequenceSeperate(batch, self.device, self.src_seq_len, self.tgt_seq_len, self.delay_embedding.dtime_transform, self.delay_embedding.len_transform, self.delay_embedding.interarrival_time_transform)
        labels = seq_obj.tgt_dtime_seqs_transformed

        mdn_params, num_predictions = self.forward(seq_obj)
        pred_dtime, pred_dtime_var = self.mdn.mean_variance(mdn_params)
        pred_q7 = self.mdn.quantile(mdn_params,q=0.7)
        pred_q9 = self.mdn.quantile(mdn_params,q=0.9)
        pred_q99 = self.mdn.quantile(mdn_params,q=0.99)
        pred_q999 = self.mdn.quantile(mdn_params,q=0.999)

        assert labels.shape == pred_dtime.shape
        assert labels.shape == pred_dtime_var.shape

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), (pred_q7, pred_q9, pred_q99, pred_q999), seq_obj.tgt_non_pad_mask, None