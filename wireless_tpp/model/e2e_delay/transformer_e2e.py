import torch
import torch.distributions as D
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, DecoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus, PositionalEncoding, FeedForwardBlock
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.utils import logger
from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise

from .emb_utils import SequenceSeperate, DelayEmbedding
from .mdn_utils import MixtureDistribution
    

class TransformerE2E(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(TransformerE2E, self).__init__(model_config)

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate

        self.tgt_seq_len = model_config.model_specs['tgt_seq_len']
        self.src_seq_len = model_config.model_specs['src_seq_len']
        self.teacher_forcing = model_config.model_specs['teacher_forcing']
        self.last_layer_mlp = model_config.model_specs['last_layer_mlp']

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

        self.include_mcs_in_tgt = model_config.model_specs['target']['include_mcs']
        self.include_mretx_in_tgt = model_config.model_specs['target']['include_mretx']
        self.include_rfailed_in_tgt = model_config.model_specs['target']['include_rfailed']
        self.PAD_TOKEN = -1.0

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
        
        # MixtureDistribution
        self.mdn = MixtureDistribution(model_config, self.device)

        if self.last_layer_mlp:
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
            self.mdn_head = nn.Linear(self.d_model, self.mdn.num_params, device = self.device)


    def encode(self, seq_obj : SequenceSeperate):
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
        embeddings = self.pos_encoder(embeddings)
        # embedding dims: [batch_size, seq_len, d_model]

        # Shift the input embeddings to the right
        shifted_embeddings = torch.zeros_like(embeddings)  # Initialize a zero tensor with the same shape as embedding
        shifted_embeddings[:, 1:, :] = embeddings[:, :-1, :]  # Shift embeddings to the right
        sh_src_embeddings = shifted_embeddings[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]
        # sh_src_embedding dims: [batch_size, src_seq_len, d_model]

        # fix the mask
        src_non_pad_mask_float = seq_obj.src_non_pad_mask.float()  # Optional if it's not already in float
        src_attention_mask = src_non_pad_mask_float.unsqueeze(1) * src_non_pad_mask_float.unsqueeze(2)  # [batch_size, src_seq_len, src_seq_len, seq_len]
        src_attention_mask = src_attention_mask == 1

        # feed in the history data to the encoder
        # [batch_size, src_seq_len, hidden_size]
        enc_output = sh_src_embeddings
        for idx, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(
                enc_output,
                mask=src_attention_mask
            )
        # enc_output dim: [batch_size, src_seq_len, d_model]

        return enc_output

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
            embeddings = self.delay_embedding(
                seq_obj.tgt_dtime_seqs[:, :idx],
                seq_obj.tgt_time_seqs[:, :idx], 
                seq_obj.tgt_interarrival_time_seqs[:, :idx], 
                seq_obj.tgt_slot_seqs[:, :idx],
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
            mdn_params = self.mdn_head(dec_out_step)
            pred_dist_step = self.mdn(mdn_params)
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
            
            new_dec_input = prev_dec_input.clone()  # VERY IMPORTANT: clone to avoid in-place modifications
            new_pad_mask = prev_pad_mask.clone()

            new_dec_input[:, idx, :] = embeddings_step.squeeze(1)
            new_pad_mask[:, idx] = seq_obj.tgt_non_pad_mask[:, idx-1]

            return new_dec_input, new_pad_mask.long()


    def forward(self, seq_obj : SequenceSeperate, forward=True):

        # teacher forcing does not work for the last layer mlp
        if self.teacher_forcing:
            assert self.last_layer_mlp == False

        is_teacher_forcing_now = self.teacher_forcing
        if not forward:
            is_teacher_forcing_now = False

        # apply embedding on the delay sequence
        enc_output = self.encode(seq_obj)
        #enc_output = torch.zeros_like(enc_output)  # Initialize a zero tensor with the same shape as enc_output
        # enc_output dim: [batch_size, src_seq_len, d_model]
 
        if self.last_layer_mlp:
            # feed the last cell's output to MLP to predict the future
            mdn_params = self.mdn_head(enc_output[:, -1:, :])
            # output is [batch_size, 1, self.tgt_seq_len*self.mdn.num_params]
            # convert it to [batch_size, self.tgt_seq_len, self.mdn.num_params]
            mdn_params = mdn_params.view(-1, self.tgt_seq_len, self.mdn.num_params)
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
            mdn_params = self.mdn_head(all_preds)
            # mdn_params: [batch_size, tgt_seq_len, self.mdn.num_params]
            num_predictions = num_predictions.item()

        return mdn_params, num_predictions
    
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