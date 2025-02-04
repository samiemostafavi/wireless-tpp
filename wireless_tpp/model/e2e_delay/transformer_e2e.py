import torch
import torch.distributions as D
from torch import nn
import random
import numpy as np

from wireless_tpp.utils import RunnerPhase
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
        
        # encoder
        self.n_encoder_heads = model_config.model_specs['encoder']['num_heads']
        self.n_encoder_layers = model_config.model_specs['encoder']['num_layers']
        self.encoder_use_residual = model_config.model_specs['encoder']['use_residual']
        self.encoder_ff_exp_rate = model_config.model_specs['encoder']['ff_exp_rate']
        logger.info(f"Encoder with {self.n_encoder_heads} heads, num_layers: {self.n_encoder_layers}, use residual: {self.encoder_use_residual}")

        # decoder
        self.last_layer_mlp = model_config.model_specs['last_layer_mlp']
        if not self.last_layer_mlp:
            self.n_decoder_self_heads = model_config.model_specs['decoder']['num_self_heads']
            self.n_decoder_cross_heads = model_config.model_specs['decoder']['num_cross_heads']
            self.n_decoder_layers = model_config.model_specs['decoder']['num_layers']
            self.decoder_use_residual = model_config.model_specs['decoder']['use_residual']
            self.decoder_ff_exp_rate = model_config.model_specs['decoder']['ff_exp_rate']
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

        self.include_prev_dtime_in_tgt = model_config.model_specs['target']['include_prev_dtime']
        self.include_slot_in_tgt = model_config.model_specs['target']['include_slot']
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
            nn.Linear(self.d_model, self.d_model * self.encoder_ff_exp_rate), # *4 is important, better than *2
            nn.GELU(), # THIS IS IMPORTANT
            nn.Dropout(p=self.dropout),
            nn.Linear(self.d_model * self.encoder_ff_exp_rate, self.d_model), # *4 is important, better than *2
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
                nn.Linear(self.d_model, self.d_model * self.decoder_ff_exp_rate), # *4 is important, better than *2
                nn.GELU(), # THIS IS IMPORTANT
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * self.decoder_ff_exp_rate, self.d_model), # *4 is important, better than *2
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


    def encode(self, embeddings_src, src_non_pad_mask):
        """Call the model

        Args:
            embeddings_src (tensor): [batch_size, src_seq_len, d_model]
        Returns:
            tensor: hidden states at event times.
        """

        # fix the mask
        src_non_pad_mask_float = src_non_pad_mask.float()  # Optional if it's not already in float
        src_attention_mask = src_non_pad_mask_float.unsqueeze(1) * src_non_pad_mask_float.unsqueeze(2)  # [batch_size, src_seq_len, src_seq_len, seq_len]
        src_attention_mask = src_attention_mask == 1

        # feed in the history data to the encoder
        # [batch_size, src_seq_len, d_model]
        enc_output = embeddings_src
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


    def get_dec_input_parallel(self, embeddings_tgt, tgt_non_pad_mask):
        # note that dec_input is a tensor of shape [batch_size, tgt_seq_len, d_model]
        # Suppose embeddings_tgt.shape = [batch_size, tgt_seq_len, d_model].
        batch_size, tgt_seq_len, d_model = embeddings_tgt.size()

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
        # copy the rest of the embeddings
        dec_input[:, 1:] = embeddings_tgt[:, 1:]
        pad_mask[:, 1:] = tgt_non_pad_mask[:, 1:]
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
            
            new_dec_input = prev_dec_input.clone()  # VERY IMPORTANT: clone to avoid in-place modifications
            new_pad_mask = prev_pad_mask.clone()

            new_dec_input[:, idx, :] = embeddings_step.squeeze(1)
            new_pad_mask[:, idx] = seq_obj.tgt_non_pad_mask[:, idx-1]

            return new_dec_input, new_pad_mask.long()


    def forward(self, seq_obj : SequenceSeperate, phase=None):

        # apply embedding on the delay sequence
        embeddings = self.embed(seq_obj)
        # embeddings dim: [batch_size, seq_len = src_len + tgt_len, d_model]

        # extract the src embedding
        embeddings_src = embeddings[:, -self.src_seq_len-self.tgt_seq_len:-self.tgt_seq_len, :]
        # src_embeddings dim: [batch_size, src_len, d_model]

        # apply embedding on the delay sequence
        enc_output = self.encode(embeddings_src, seq_obj.src_non_pad_mask)
        # enc_output dim: [batch_size, src_seq_len, d_model]
 
        if self.last_layer_mlp:
            # feed the last cell's output to MLP to predict the future
            mdn_params = self.mdn_head(enc_output[:, -1:, :])
            # output is [batch_size, 1, self.tgt_seq_len*self.mdn.num_params]
            # convert it to [batch_size, self.tgt_seq_len, self.mdn.num_params]
            mdn_params = mdn_params.view(-1, self.tgt_seq_len, self.mdn.num_params)
            num_predictions = seq_obj.tgt_non_pad_mask.sum(axis=0)
        else:     
            # use decoder to predict the target sequence
            if self.include_prev_dtime_in_tgt:
                # autoregressive decoding
                # We'll store predictions for each time step
                all_preds = []
                num_predictions = torch.zeros(self.tgt_seq_len)
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
                    num_predictions[idx] = tgt_pad_mask[:, idx].sum()

                # and feed the results into a final linear to get distribution parameters.
                # 5) Convert all_preds => [batch_size, tgt_seq_len, d_model]
                all_preds = torch.stack(all_preds, dim=1)
                mdn_params = self.mdn_head(all_preds)
                # mdn_params: [batch_size, tgt_seq_len, self.mdn.num_params]
            else:
                # parallel decoding
                # predicts the entire decoder sequence at once
                # the output of the previous step will be paddings
                # extract the src embedding
                embeddings_tgt = embeddings[:, -self.src_seq_len:, :]
                # embeddings_tgt dim: [batch_size, tgt_len, d_model]

                dec_input, tgt_pad_mask = self.get_dec_input_parallel(embeddings_tgt, seq_obj.tgt_non_pad_mask)

                # 4) Pass into the decoder
                # dec_input: [batch_size, tgt_seq_len]
                # enc_output: [batch_size, src_seq_len, d_model]
                # src_pad_mask: [batch_size, src_seq_len] mask for encoder outputs (e.g. padding mask)
                # tgt_pad_mask_tf: [batch_size, tgt_seq_len, tgt_seq_len] mask for decoder inputs (e.g. padding mask)
                # dec_out => [batch_size, tgt_seq_len, d_model]
                dec_out = self.decode(
                    dec_input_emb=dec_input,
                    enc_output=enc_output,
                    src_pad_mask=seq_obj.src_non_pad_mask.float(),
                    tgt_pad_mask=tgt_pad_mask
                )
                num_predictions = tgt_pad_mask.sum(axis=0)
                mdn_params = self.mdn_head(dec_out)


        return mdn_params, num_predictions
    
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