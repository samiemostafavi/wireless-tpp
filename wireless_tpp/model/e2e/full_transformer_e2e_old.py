import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise


class FullTransformerE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(FullTransformerE2E, self).__init__(model_config)

        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)
        self.mean_len = model_config.model_specs.get("mean_len", 0.0)
        self.std_len = model_config.model_specs.get("std_len", 1.0)
        logger.info(f"FullTransformerE2E loading mean and std of dtime: {self.mean_dtime}, {self.std_dtime}")
        self.dtime_transform = D.AffineTransform(loc=self.mean_dtime, scale=self.std_dtime)
        logger.info(f"FullTransformerE2E loading mean and std of len: {self.mean_len}, {self.std_len}")
        self.len_hist_transform = D.AffineTransform(loc=self.mean_len, scale=self.std_len)

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
        
        self.use_norm = model_config.use_ln
        self.n_encoder_heads = model_config.model_specs['encoder']['num_heads']
        self.n_encoder_layers = model_config.model_specs['encoder']['num_layers']

        self.n_decoder_self_heads = model_config.model_specs['decoder']['num_self_heads']
        self.n_decoder_cross_heads = model_config.model_specs['decoder']['num_cross_heads']
        self.n_decoder_layers = model_config.model_specs['decoder']['num_layers']

        self.tgt_seq_len = model_config.model_specs['tgt_seq_len']
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

        self.num_event_types_pad = 7 # should be 4
        self.pad_token_id = 6
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

        self.dec_input_embed = nn.Linear(1, self.d_model)
        self.dec_input_pos_encoder = PositionalEncoding(self.d_model, self.dropout, device=self.device)

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
                use_residual=False,
                feed_forward=self.feed_forward_encoder,
                dropout=self.dropout
            ) for _ in range(self.n_encoder_layers)])
        
        # decoder MLP layer
        self.feed_forward_decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
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
                    use_residual=False,
                    dropout=self.dropout
            ) for _ in range(self.n_decoder_layers)])
        
        # prediction linear layer
        self.dtime_linear = nn.Linear(self.d_model, 3 * self.num_mix_components_dtime)


    def decode(self, dec_input, enc_output, src_pad_mask, tgt_pad_mask):
        # dec_input is supposed to be delay values: [batch_size, tgt_seq_len]
        # enc_output: [batch_size, src_seq_len, d_model]
        # src_inp_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
        # tgt_inp_mask: [batch_size, tgt_seq_len] subsequent mask for decoder to prevent seeing future tokens

        dec_input = dec_input.unsqueeze(-1).float() # -> [batch_size, tgt_seq_len, 1]
        dec_input_emb = self.dec_input_embed(dec_input)  # -> [batch_size, tgt_seq_len, d_model]
        dec_input_emb = self.dec_input_pos_encoder(dec_input_emb) # -> [batch_size, tgt_seq_len, d_model]

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
            if idx > 0:
                dec_output += dec_input_emb
            dec_output = dec_layer(
                dec_output,  # [batch_size, tgt_seq_len, d_model] is needed
                enc_output, # [batch_size, src_seq_len, d_model] is needed
                tgt_mask=tgt_mask,  # [batch_size, tgt_seq_len, tgt_seq_len] Mask for the target sequence (usually for preventing attention to future tokens)
                mask_2d=mask_2d # [batch_size, tgt_seq_len, src_seq_len] Mask for the cross attention (e.g., padding mask)
            )
        return dec_output


    def encode(self, history_seq_obj):
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
            batch_non_pad_mask, attention_mask = history_seq_obj.get_all()

        # only linear ones need unsqueeze
        # convert type_seqs to int type for embedding
        type_seqs = type_seqs.long()
        slot_seqs = slot_seqs.long()
        mcs_seqs = mcs_seqs.long()
        mretx_seqs = mretx_seqs.long()
        rfailed_seqs = rfailed_seqs.long()
        num_rbs_seqs = num_rbs_seqs.long()

        len_seqs = self.len_hist_transform.inv(len_seqs_transformed) # apply inverse transform to len
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

        # encoder_mask: shape [batch_size, src_seq_len, src_seq_len]
        # 1 => masked, 0 => not masked

        # We can say: "A source token is considered padded if the entire row is masked."
        # or if the diagonal is masked. It depends on how you built it.

        # One common approach:
        src_pad_mask_1d = (attention_mask.sum(dim=-1) == self.src_seq_len)  
        # shape [batch_size, src_seq_len], True where row was entirely masked => it's a PAD token
        # or you might want != 0, depending on how you built it

        # Then invert or convert to 1/0 for "valid vs pad":
        # e.g. 1 => valid, 0 => pad
        src_pad_mask_1d = (~src_pad_mask_1d).long()  # shape [batch_size, src_seq_len] 

        return enc_output, src_pad_mask_1d


    def get_pred_distribution(self, dec_out) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            dec_out (tensor): [batch_size, tgt_seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution.
        """
        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.dtime_linear(dec_out)

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

        history_seq_obj = HistorySequence(batch, self.src_seq_len, self.tgt_seq_len)
        target_seq_obj = TargetSequence(batch, self.src_seq_len, self.tgt_seq_len, self.device, self.dtime_transform)

        # 1. encode the history
        # enc_out: [batch_size, src_seq_len, hidden_size]
        # src_mask: [batch_size, src_seq_len]
        enc_out, src_pad_mask = self.encode(history_seq_obj)

        # We'll store predictions for each time step
        all_preds = []
        num_predictions = 0
        # 3) Auto-regressive decoding
        #    for each idx in [0..(tgt_seq_len-1)], feed partial dec_input
        for idx in range(self.tgt_seq_len):
            # dec_input => [batch_size, tgt_seq_len], partial sequence up to idx-1
            # tgt_mask => [batch_size, tgt_seq_len], 1=real token, 0=pad token
            dec_input, tgt_pad_mask = target_seq_obj.get_dec_input(idx) 

            #print(src_pad_mask.detach().cpu().numpy())
            #print(tgt_pad_mask.detach().cpu().numpy())
            #print(dec_input.detach().cpu().numpy())
            #input()

            # 4) Pass into the decoder
            # dec_input: [batch_size, tgt_seq_len]
            # enc_output: [batch_size, src_seq_len, d_model]
            # src_pad_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
            # tgt_pad_mask: [batch_size, tgt_seq_len] mask for decoder inputs (e.g. padding mask)
            # dec_out => [batch_size, tgt_seq_len, d_model]
            dec_out = self.decode(
                dec_input=dec_input,
                enc_output=enc_out,
                src_pad_mask=src_pad_mask,
                tgt_pad_mask=tgt_pad_mask
            )
            # we take the last position (i.e. dec_out[:, idx, :]) for prediction
            step_out = dec_out[:, idx, :] # shape [batch_size, d_model]
            all_preds.append(step_out)
            num_predictions += tgt_pad_mask[:, idx].sum()

        # and feed the results into a final linear to get distribution parameters.
        # 5) Convert all_preds => [batch_size, tgt_seq_len, d_model]
        all_preds = torch.stack(all_preds, dim=1)
        pred_dist = self.get_pred_distribution(all_preds)
        # result: [batch_size, tgt_seq_len]
        
        # 6) Compute negative log-likelihood vs. the ground-truth times
        labels = target_seq_obj.dtime_seqs_transformed  # [batch_size, tgt_seq_len]
        assert labels.shape == pred_dist.mean.shape

        dtime_ll = pred_dist.log_prob(labels)
        dtime_loss = -dtime_ll.sum()
        
        return dtime_loss, num_predictions.item(), None, None


    def predict_mean_variance(self, batch, forward=False):
        history_seq_obj = HistorySequence(batch, self.src_seq_len, self.tgt_seq_len)
        target_seq_obj = TargetSequence(batch, self.src_seq_len, self.tgt_seq_len, self.device, self.dtime_transform)

        # 1. encode the history
        # enc_out: [batch_size, src_seq_len, hidden_size]
        # src_mask: [batch_size, src_seq_len]
        enc_out, src_pad_mask = self.encode(history_seq_obj)

        # Prepare lists to store predictions
        all_means = []
        all_vars = []

        # We'll keep track of predictions at each step 
        # The shape is [batch_size], storing the last predicted dtimes for each item in the batch.
        # Initially, we can store zeros (or your SOS_TOKEN) for the "previous" time.
        #last_predictions = torch.zeros(
        #    enc_out.size(0),
        #    device=self.device,
        #    dtype=torch.float
        #)
        batch_size = enc_out.size(0)
        pred_seq = torch.full(
            (batch_size, self.tgt_seq_len),
            fill_value=self.PAD_TOKEN,
            device=self.device
        )
        num_predictions = 0
        # 3) Auto-regressive decoding for each step in [0..tgt_seq_len-1]
        for idx in range(self.tgt_seq_len):

            # (Pure inference) We'll feed *our own predictions* back.
            # We create dec_input ourselves by storing predictions in the appropriate slot.
            #dec_input, tgt_pad_mask = self.get_dec_input_inference(idx, last_predictions, self.tgt_seq_len)
            # get_dec_input_inference places the predicted values up to idx-1 in dec_input. 
            # e.g. set dec_input[:, idx] = last_predictions
            # output dims: [batch_size, tgt_seq_len]

            # 1) Build dec_input from scratch
            dec_input = torch.full(
                (batch_size, self.tgt_seq_len), 
                fill_value=self.PAD_TOKEN,
                device=self.device
            )
            dec_input[:, 0] = self.SOS_TOKEN

            # 2) If idx > 0, fill in the previously predicted steps [0..idx-1]
            #    into positions [1..idx]
            if idx > 0:
                dec_input[:, 1:idx+1] = self.dtime_transform.inv(pred_seq[:, :idx])
        
            # 3) Build mask
            tgt_pad_mask = (dec_input != self.PAD_TOKEN).long()

            # 4) Decode
            dec_out = self.decode(
                dec_input=dec_input,        # [batch_size, tgt_seq_len] (will be embedded internally)
                enc_output=enc_out,         # [batch_size, src_seq_len, d_model]
                src_pad_mask=src_pad_mask,  # [batch_size, src_seq_len]
                tgt_pad_mask=tgt_pad_mask   # [batch_size, tgt_seq_len]
            )
            # dec_out: [batch_size, tgt_seq_len, d_model]

            # 5) Take the hidden state at position idx
            step_out = dec_out[:, idx, :]  # [batch_size, d_model]


            # fix step_out shape to [batch_size, 1, d_model]
            step_out = step_out.unsqueeze(1)

            # 6) Convert it to distribution parameters
            #    We'll assume self.get_pred_distribution(...) returns a distribution with .mean and .variance
            step_dist = self.get_pred_distribution(step_out)  # shape [batch_size], or [batch_size, ...]

            # Extract the mean/variance
            step_mean = step_dist.mean[...,0]  # [batch_size]
            step_var  = step_dist.variance[...,0] # [batch_size]

            # 5) Store the newly predicted times in pred_seq
            pred_seq[:, idx] = step_mean

            # 7) Save them for the entire sequence
            all_means.append(step_mean)
            all_vars.append(step_var)

            num_predictions += tgt_pad_mask[:, idx].sum()

        # 8) Stack the results: [batch_size, tgt_seq_len]
        pred_dtime = torch.stack(all_means, dim=1)
        pred_dtime_var  = torch.stack(all_vars, dim=1)

        labels = target_seq_obj.dtime_seqs_transformed  # [batch_size, tgt_seq_len]
        assert labels.shape == pred_dtime.shape
        assert labels.shape == pred_dtime_var.shape

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), None, num_predictions.item()


    def predict_probabilities(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        slot_seqs = slot_seqs[:, -1 -self.his_len:]
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        type_seqs = type_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            slot_seqs[:, :-1],
            len_seqs_transformed[:, :-1], 
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
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

            return (dtimes_logprob_pred, lens_logprob_pred), (dtime_seqs_transformed, time_seqs, type_seqs, slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events
        else:
            pred_joint_dist = self.joint_distribution(enc_out)
            
            # Step 1: Create 1D linspace for each dimension
            sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
            sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
            num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
            sample_len_min = prediction_config['probability_generation']['sample_len_min']
            sample_len_max = prediction_config['probability_generation']['sample_len_max']
            num_steps_len = prediction_config['probability_generation']['num_steps_len']
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

            return (joint_logpdfs_pred, ), (dtime_seqs_transformed, time_seqs, type_seqs, slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events

        
    def generate_samples(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        slot_seqs = slot_seqs[:, -1 -self.his_len:]
        len_seqs_transformed = len_seqs_transformed[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        type_seqs = type_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs_transformed = dtime_seqs_transformed[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            slot_seqs[:, :-1],
            len_seqs_transformed[:, :-1], 
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
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
        len_samples = pred_len_dist.sample((prediction_config['num_samples_len'],))

        return (dtimes_samples, len_samples), (dtime_seqs_transformed, time_seqs, type_seqs, slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events
    



class HistorySequence():
    # the data sequence is divided into two parts: history and target
    # [:, -self.tgt_seq_len:] is the target sequence with the length of self.tgt_seq_len
    # target sequence consists of only departure events type = max_type
    # [:, -2-self.src_seq_len:-self.tgt_seq_len] is the history sequence with the length: self.src_seq_len
    # history sequence does not include departure events
    def __init__(self, batch, src_seq_len, tgt_seq_len):
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.src_seq_len events in the history
        self.slot_seqs = slot_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.len_seqs_transformed = len_seqs_transformed[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.mcs_seqs = mcs_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.mretx_seqs = mretx_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.rfailed_seqs = rfailed_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.num_rbs_seqs = num_rbs_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.time_seqs = time_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.type_seqs = type_seqs[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.attention_mask = attention_mask[:, -src_seq_len-tgt_seq_len:-tgt_seq_len, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.dtime_seqs_transformed = dtime_seqs_transformed[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
        self.batch_non_pad_mask = batch_non_pad_mask[:, -src_seq_len-tgt_seq_len:-tgt_seq_len]
    def get_all(self):
        return self.slot_seqs, self.len_seqs_transformed, self.mcs_seqs, self.mretx_seqs, self.rfailed_seqs, self.num_rbs_seqs, self.time_seqs, self.dtime_seqs_transformed, self.type_seqs, self.batch_non_pad_mask, self.attention_mask
    
class TargetSequence():
    # the data sequence is divided into two parts: history and target
    # [:, -self.tgt_seq_len:] is the target sequence with the length of self.tgt_seq_len
    # target sequence consists of only departure events type = max_type
    # in departure events, dtime is not the time since last event, it is the packet delay
    # [:, -2-self.src_seq_len:-self.tgt_seq_len] is the history sequence with the length: self.src_seq_len
    def __init__(self, batch, src_seq_len, tgt_seq_len, device, dtime_transform):
        self.SOS_TOKEN = 0.0
        self.PAD_TOKEN = -1.0
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.device = device
        self.dtime_transform = dtime_transform
        slot_seqs, len_seqs_transformed, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs, time_seqs, dtime_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.tgt_seq_len events in the target
        self.dtime_seqs_transformed = dtime_seqs_transformed[:, -self.tgt_seq_len:]
        self.dtime_seqs = self.dtime_transform.inv(self.dtime_seqs_transformed)
        self.attention_mask = attention_mask[:, -self.tgt_seq_len:, -self.tgt_seq_len:]
        self.batch_non_pad_mask = batch_non_pad_mask[:, -self.tgt_seq_len:]
    
    def get_dec_input(self, idx : int):
        """
        Construct partial decoder input for time-step `idx`.

        Args:
            idx (int): up to which index in dtime_seqs we want to feed the decoder.
                    For example, if idx=0, we only feed [SOS, PAD, PAD, ...].
                    If idx=1, we feed [SOS, dtime_seqs[:,0], PAD, PAD, ...], etc.

        Returns:
            dec_input (Tensor): shape [batch_size, tgt_seq_len].
                                At position 0: SOS token.
                                Positions [1..idx] : previously observed deltas.
                                Positions [idx+1..end] : padding.

            pad_mask: [batch_size, tgt_seq_len], where 1 indicates "real token",
                    and 0 indicates "pad token".
        """
        # Suppose self.dtime_seqs.shape = [batch_size, tgt_seq_len].
        batch_size, tgt_seq_len = self.dtime_seqs.size()

        # Create an all-PAD tensor
        dec_input = torch.full(
            (batch_size, tgt_seq_len), 
            fill_value=self.PAD_TOKEN,
            device=self.device
        )
        
        # Put the SOS token at position 0
        dec_input[:, 0] = self.SOS_TOKEN
        
        # Copy ground-truth dtimes up to idx-1 into positions [1..idx]
        # (Note: if idx=0, this does nothing.)
        if idx > 0:
            dec_input[:, 1:idx+1] = self.dtime_seqs[:, :idx]

        # Mark positions as 1 if not PAD_TOKEN, else 0
        pad_mask = (dec_input != self.PAD_TOKEN).long()
        
        return dec_input, pad_mask

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

    def get_all(self):
        return self.slot_seqs, self.len_seqs_transformed, self.mcs_seqs, self.mretx_seqs, self.rfailed_seqs, self.num_rbs_seqs, self.time_seqs, self.dtime_seqs_transformed, self.type_seqs, self.batch_non_pad_mask
    
    def get_dec_input(self, idx : int):
        """
        Construct partial decoder input for time-step `idx`.

        Args:
            idx (int): up to which index in dtime_seqs we want to feed the decoder.
                    For example, if idx=0, we only feed [SOS, PAD, PAD, ...].
                    If idx=1, we feed [SOS, dtime_seqs[:,0], PAD, PAD, ...], etc.

        Returns:
            dec_input (Tensor): shape [batch_size, tgt_seq_len].
                                At position 0: SOS token.
                                Positions [1..idx] : previously observed deltas.
                                Positions [idx+1..end] : padding.

            pad_mask: [batch_size, tgt_seq_len], where 1 indicates "real token",
                    and 0 indicates "pad token".
        """
        # Suppose self.dtime_seqs.shape = [batch_size, tgt_seq_len].
        batch_size, seq_len = self.dtime_seqs.size()

        # Create an all-PAD tensor
        dec_input = torch.full(
            (batch_size, seq_len), 
            fill_value=self.PAD_TOKEN,
            device=self.device
        )

        pad_mask = torch.full(
            (batch_size, seq_len), 
            fill_value=False,
            device=self.device
        )
        
        # Put the SOS token at position 0
        dec_input[:, 0] = self.SOS_TOKEN
        pad_mask[:, 0] = True

        # Copy ground-truth dtimes up to idx-1 into positions [1..idx]
        # (Note: if idx=0, this does nothing.)
        if idx > 0:
            dec_input[:, 1:idx+1] = self.dtime_seqs[:, :idx]
            pad_mask[:, 1:idx+1] = self.batch_non_pad_mask[:, :idx]
        
        return dec_input, pad_mask.long()


class FullTransformerE2ENew(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    This is the arrival sequence based model.
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(FullTransformerE2ENew, self).__init__(model_config)

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
        
        self.use_norm = model_config.use_ln
        self.n_encoder_heads = model_config.model_specs['encoder']['num_heads']
        self.n_encoder_layers = model_config.model_specs['encoder']['num_layers']

        self.n_decoder_self_heads = model_config.model_specs['decoder']['num_self_heads']
        self.n_decoder_cross_heads = model_config.model_specs['decoder']['num_cross_heads']
        self.n_decoder_layers = model_config.model_specs['decoder']['num_layers']

        self.tgt_seq_len = model_config.model_specs['tgt_seq_len']
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

        self.dec_input_embed = nn.Linear(1, self.d_model)
        self.dec_input_pos_encoder = PositionalEncoding(self.d_model, self.dropout, device=self.device)

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
                use_residual=False,
                feed_forward=self.feed_forward_encoder,
                dropout=self.dropout
            ) for _ in range(self.n_encoder_layers)])
        
        # decoder MLP layer
        self.feed_forward_decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
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
                    use_residual=False,
                    dropout=self.dropout
            ) for _ in range(self.n_decoder_layers)])
        
        # prediction linear layer
        self.dtime_linear = nn.Linear(self.d_model, 3 * self.num_mix_components_dtime)


    def decode(self, dec_input, enc_output, src_pad_mask, tgt_pad_mask):
        # dec_input is supposed to be delay values: [batch_size, seq_len]
        # enc_output: [batch_size, seq_len, d_model]
        # src_inp_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
        # tgt_inp_mask: [batch_size, tgt_seq_len] subsequent mask for decoder to prevent seeing future tokens

        dec_input = dec_input.unsqueeze(-1).float() # -> [batch_size, tgt_seq_len, 1]
        dec_input_emb = self.dec_input_embed(dec_input)  # -> [batch_size, tgt_seq_len, d_model]
        dec_input_emb = self.dec_input_pos_encoder(dec_input_emb) # -> [batch_size, tgt_seq_len, d_model]

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
            if idx > 0:
                dec_output += dec_input_emb
            dec_output = dec_layer(
                dec_output,  # [batch_size, tgt_seq_len, d_model] is needed
                enc_output, # [batch_size, src_seq_len, d_model] is needed
                tgt_mask=tgt_mask,  # [batch_size, tgt_seq_len, tgt_seq_len] Mask for the target sequence (usually for preventing attention to future tokens)
                mask_2d=mask_2d # [batch_size, tgt_seq_len, src_seq_len] Mask for the cross attention (e.g., padding mask)
            )
        return dec_output


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
            non_pad_mask = seq_obj.get_all()

        # fix the mask
        non_pad_mask_float = non_pad_mask.float()  # Optional if it's not already in float
        attention_mask = non_pad_mask_float.unsqueeze(1) * non_pad_mask_float.unsqueeze(2)  # [batch_size, seq_len, seq_len]
        attention_mask = attention_mask == 1

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


    def get_pred_distribution(self, dec_out) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:
            dec_out (tensor): [batch_size, tgt_seq_len, hidden_size], hidden states at event times.

        Returns:
            NormalMixtureDistribution: delta time distribution.
        """
        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.dtime_linear(dec_out)

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

        # We'll store predictions for each time step
        all_preds = []
        num_predictions = 0
        # 3) Auto-regressive decoding
        #    for each idx in [0..(seq_len-1)], feed partial dec_input
        for idx in range(self.seq_len):
            # dec_input => [batch_size, seq_len], partial sequence up to idx-1
            # tgt_mask => [batch_size, seq_len], 1=real token, 0=pad token
            dec_input, tgt_pad_mask = seq_obj.get_dec_input(idx) 

            # 4) Pass into the decoder
            # dec_input: [batch_size, tgt_seq_len]
            # enc_output: [batch_size, src_seq_len, d_model]
            # src_pad_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
            # tgt_pad_mask: [batch_size, tgt_seq_len] mask for decoder inputs (e.g. padding mask)
            # dec_out => [batch_size, tgt_seq_len, d_model]
            dec_out = self.decode(
                dec_input=dec_input,
                enc_output=enc_out,
                src_pad_mask=src_pad_mask,
                tgt_pad_mask=tgt_pad_mask
            )
            # we take the last position (i.e. dec_out[:, idx, :]) for prediction
            step_out = dec_out[:, idx, :] # shape [batch_size, d_model]
            all_preds.append(step_out)
            num_predictions += tgt_pad_mask[:, idx].sum()

        # and feed the results into a final linear to get distribution parameters.
        # 5) Convert all_preds => [batch_size, seq_len, d_model]
        all_preds = torch.stack(all_preds, dim=1)
        pred_dist = self.get_pred_distribution(all_preds)
        # result: [batch_size, seq_len]
        
        # 6) Compute negative log-likelihood vs. the ground-truth times
        labels = seq_obj.dtime_seqs_transformed  # [batch_size, seq_len]
        assert labels.shape == pred_dist.mean.shape

        dtime_ll = pred_dist.log_prob(labels) * seq_obj.batch_non_pad_mask.long()
        dtime_loss = -dtime_ll.sum()
        
        return dtime_loss, num_predictions.item(), None, None


    def predict_mean_variance(self, batch, forward=False):
        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform)
        self.seq_len = seq_obj.dtime_seqs.size(1)
        batch_size = seq_obj.dtime_seqs.size(0)

        # 1. encode the history
        # enc_out: [batch_size, seq_len, hidden_size]
        # src_mask: [batch_size, seq_len]
        enc_out, src_pad_mask = self.encode(seq_obj)

        # Prepare lists to store predictions
        all_means = []
        all_vars = []

        # We'll keep track of predictions at each step 
        # The shape is [batch_size], storing the last predicted dtimes for each item in the batch.
        # Initially, we can store zeros (or your SOS_TOKEN) for the "previous" time.
        #last_predictions = torch.zeros(
        #    enc_out.size(0),
        #    device=self.device,
        #    dtype=torch.float
        #)
        pred_seq = torch.full(
            (batch_size, self.seq_len),
            fill_value=self.PAD_TOKEN,
            device=self.device
        )
        num_predictions = 0
        # 3) Auto-regressive decoding for each step in [0..tgt_seq_len-1]
        for idx in range(self.seq_len):

            # (Pure inference) We'll feed *our own predictions* back.
            # We create dec_input ourselves by storing predictions in the appropriate slot.
            #dec_input, tgt_pad_mask = self.get_dec_input_inference(idx, last_predictions, self.tgt_seq_len)
            # get_dec_input_inference places the predicted values up to idx-1 in dec_input. 
            # e.g. set dec_input[:, idx] = last_predictions
            # output dims: [batch_size, tgt_seq_len]

            # 1) Build dec_input from scratch
            dec_input = torch.full(
                (batch_size, self.seq_len), 
                fill_value=self.PAD_TOKEN,
                device=self.device
            )
            tgt_pad_mask = torch.full(
                (batch_size, self.seq_len), 
                fill_value=False,
                device=self.device
            )
            dec_input[:, 0] = self.SOS_TOKEN
            tgt_pad_mask[:, 0] = True
            
            # 2) If idx > 0, fill in the previously predicted steps [0..idx-1]
            #    into positions [1..idx]
            if idx > 0:
                dec_input[:, 1:idx+1] = self.dtime_transform.inv(pred_seq[:, :idx])
                tgt_pad_mask[:, 1:idx+1] =  seq_obj.batch_non_pad_mask[:, :idx]

            # 3) Decode
            dec_out = self.decode(
                dec_input=dec_input,                # [batch_size, seq_len] (will be embedded internally)
                enc_output=enc_out,                 # [batch_size, seq_len, d_model]
                src_pad_mask=src_pad_mask,          # [batch_size, seq_len]
                tgt_pad_mask=tgt_pad_mask.long()    # [batch_size, seq_len]
            )
            # dec_out: [batch_size, seq_len, d_model]

            # 5) Take the hidden state at position idx
            step_out = dec_out[:, idx, :]  # [batch_size, d_model]


            # fix step_out shape to [batch_size, 1, d_model]
            step_out = step_out.unsqueeze(1)

            # 6) Convert it to distribution parameters
            #    We'll assume self.get_pred_distribution(...) returns a distribution with .mean and .variance
            step_dist = self.get_pred_distribution(step_out)  # shape [batch_size], or [batch_size, ...]

            # Extract the mean/variance
            step_mean = step_dist.mean[...,0]  # [batch_size]
            step_var  = step_dist.variance[...,0] # [batch_size]

            # 5) Store the newly predicted times in pred_seq
            pred_seq[:, idx] = step_mean

            # 7) Save them for the entire sequence
            all_means.append(step_mean)
            all_vars.append(step_var)

            num_predictions += tgt_pad_mask[:, idx].sum()

        # 8) Stack the results: [batch_size, tgt_seq_len]
        pred_dtime = torch.stack(all_means, dim=1)
        pred_dtime_var  = torch.stack(all_vars, dim=1)

        labels = seq_obj.dtime_seqs_transformed  # [batch_size, tgt_seq_len]
        assert labels.shape == pred_dtime.shape
        assert labels.shape == pred_dtime_var.shape

        return (pred_dtime,pred_dtime_var), (None,None), (labels, None), seq_obj.batch_non_pad_mask, num_predictions.item()
    

class FullTransformerE2ENewEnc(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(FullTransformerE2ENewEnc, self).__init__(model_config)

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
        
        self.use_norm = model_config.use_ln
        self.n_encoder_heads = model_config.model_specs['encoder']['num_heads']
        self.n_encoder_layers = model_config.model_specs['encoder']['num_layers']

        self.n_decoder_self_heads = model_config.model_specs['decoder']['num_self_heads']
        self.n_decoder_cross_heads = model_config.model_specs['decoder']['num_cross_heads']
        self.n_decoder_layers = model_config.model_specs['decoder']['num_layers']

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
                use_residual=False,
                feed_forward=self.feed_forward_encoder,
                dropout=self.dropout
            ) for _ in range(self.n_encoder_layers)])
        
        # prediction linear layer
        self.dtime_linear = nn.Linear(self.d_model, 3 * self.num_mix_components_dtime)

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
            non_pad_mask = seq_obj.get_all()

        # fix the mask
        non_pad_mask_float = non_pad_mask.float()  # Optional if it's not already in float
        attention_mask = non_pad_mask_float.unsqueeze(1) * non_pad_mask_float.unsqueeze(2)  # [batch_size, seq_len, seq_len]
        attention_mask = attention_mask == 1

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
            enc_out (tensor): [batch_size, 1, hidden_size], hidden states at event times.

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

        dtime_ll = pred_dist.log_prob(labels) * seq_obj.batch_non_pad_mask.long()
        dtime_loss = -dtime_ll.sum()
        
        return dtime_loss, num_predictions.item(), None, None


    def predict_mean_variance(self, batch, forward=False):
        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform)
        self.seq_len = seq_obj.dtime_seqs.size(1)
        batch_size = seq_obj.dtime_seqs.size(0)

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
