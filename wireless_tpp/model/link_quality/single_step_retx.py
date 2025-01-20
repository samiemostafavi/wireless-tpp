import torch
from torch.distributions import Categorical
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, MultiHeadAttention, TimePositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel


class SingleStepRETX(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(SingleStepRETX, self).__init__(model_config)

        self.concat_embeddings = model_config.model_specs['embeddings']['concat']
        self.concat_conditions = model_config.model_specs['conditions']['concat']

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.his_len = model_config.model_specs['history']['length']
        self.mretx_emb_dim = model_config.model_specs['embeddings']['mretx_emb_dim'] if self.concat_embeddings else self.d_model

        self.include_num_rbs = model_config.model_specs['history']['include_num_rbs']
        self.num_rbs_emb_dim = model_config.model_specs['embeddings']['num_rbs_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_mcs = model_config.model_specs['history']['include_mcs']
        self.mcs_emb_dim = model_config.model_specs['embeddings']['mcs_emb_dim'] if self.concat_embeddings else self.d_model

        self.include_rfailed = model_config.model_specs['history']['include_rfailed']
        self.rfailed_emb_dim = model_config.model_specs['embeddings']['rfailed_emb_dim'] if self.concat_embeddings else self.d_model

        self.include_num_rbs_cond = model_config.model_specs['conditions']['include_num_rbs']
        if self.concat_conditions == 'concat_all':
            self.num_rbs_cond_emb_dim = model_config.model_specs['conditions']['num_rbs_emb_dim']
        elif self.concat_conditions == 'sum_conds':
            self.conds_emb_dim = model_config.model_specs['conditions']['cond_dim']
            self.num_rbs_cond_emb_dim = self.conds_emb_dim
        elif self.concat_conditions == 'sum_all':
            self.num_rbs_cond_emb_dim = self.d_model

        self.include_mcs_cond = model_config.model_specs['conditions']['include_mcs']
        if self.concat_conditions == 'concat_all':
            self.mcs_cond_emb_dim = model_config.model_specs['conditions']['mcs_emb_dim']
        elif self.concat_conditions == 'sum_conds':
            self.conds_emb_dim = model_config.model_specs['conditions']['cond_dim']
            self.mcs_cond_emb_dim = self.conds_emb_dim
        elif self.concat_conditions == 'sum_all':
            self.mcs_cond_emb_dim = self.d_model
        else:
            raise ValueError(f"Invalid concatenation type: {self.concat_conditions}. Choose from ['concat_all', 'sum_conds', 'sum_all'].")

        self.pooling = model_config.model_specs['pooling']['type']

        if self.concat_embeddings:
            # size of time embedding is self.d_model minues the total size of the other embeddings
            self.time_emb_size = self.d_model - (
                int(self.include_num_rbs)*self.num_rbs_emb_dim + \
                int(self.include_mcs)*self.mcs_emb_dim + \
                int(self.include_mretx)*self.mretx_emb_dim + \
                int(self.include_rfailed)*self.rfailed_emb_dim
            )
        else:
            self.time_emb_size = self.d_model

        if self.concat_conditions == 'concat_all':
            self.cond_hidden_size = self.d_model + self.num_rbs_cond_emb_dim + self.mcs_cond_emb_dim
        elif self.concat_conditions == 'sum_conds':
            self.cond_hidden_size = self.d_model + self.conds_emb_dim
        elif self.concat_conditions == 'sum_all':
            self.cond_hidden_size = self.d_model
        else:
            raise ValueError(f"Invalid concatenation type: {self.concat_conditions}. Choose from ['concat_all', 'sum_conds', 'sum_all'].")
        
        self.use_norm = model_config.use_ln
        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate
        
        # History embedding configurations
        # num rbs embedding
        self.num_rbs_types = 107  # slot indices: 0 to 19 (20 types), and padding token
        self.num_rbs_pad_id = 106
        # mcs embedding
        self.num_mcs_types = 30  # MCS indices: 0 to 28 (29 types), and padding token
        self.mcs_pad_id = 29
        # retransmissions embedding
        self.num_mretx_types = 5  # retransmission indices: 0 to 3 (4 types), and padding token
        self.mretx_pad_id = 4
        # rlc failed embedding
        self.num_rfailed_types = 3  # failed attempt indices: 0 and 1 (2 types), and padding token
        self.rfailed_pad_id = 2

        #self.num_event_types_pad = ?

        # Embedding layers defenitions
        # temporal encoding
        self.layer_temporal_encoding = TimePositionalEncoding(
            self.time_emb_size, device=self.device
        )
        # retransmissions encoding
        self.layer_mretx_emb = nn.Embedding(
            self.num_mretx_types,
            self.mretx_emb_dim,
            padding_idx=self.mretx_pad_id,
            device=self.device
        )
        if self.include_num_rbs:
            # slot number encoding
            self.layer_num_rbs_emb = nn.Embedding(
                self.num_rbs_types,
                self.num_rbs_emb_dim,
                padding_idx=self.num_rbs_pad_id,
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
        if self.include_rfailed:
            # failed attempt encoding
            self.layer_rfailed_emb = nn.Embedding(
                self.num_rfailed_types,
                self.rfailed_emb_dim, 
                padding_idx=self.rfailed_pad_id,
                device=self.device
            )
        if self.include_num_rbs_cond:
            # slot number encoding
            self.layer_num_rbs_cond_emb = nn.Embedding(
                self.num_rbs_types,
                self.num_rbs_cond_emb_dim,
                padding_idx=self.num_rbs_pad_id,
                device=self.device
            )
        if self.include_mcs_cond:
            # mcs encoding
            self.layer_mcs_cond_emb = nn.Embedding(
                self.num_mcs_types,
                self.mcs_cond_emb_dim,
                padding_idx=self.mcs_pad_id,
                device=self.device
            )

        # MLP layer (self.feed_forward) without condition
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        # Transformer layers (self.stack_layers)
        self.stack_layers = nn.ModuleList(
            [EncoderLayer(
                self.d_model,
                MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                   output_linear=False),
                use_residual=False,
                feed_forward=self.feed_forward,
                dropout=self.dropout
            ) for _ in range(self.n_layers)])
        
        # mretx prediction linear layer
        self.mretx_linear = nn.Linear(self.cond_hidden_size, self.num_mretx_types)

        # rfailed prediction linear layer
        self.rfailed_linear = nn.Linear(self.cond_hidden_size, self.num_rfailed_types)


    def forward_cond(self, enc_out, num_rbs_seqs_label, mcs_seqs_label):
        num_rbs_seqs_label = num_rbs_seqs_label.long()
        mcs_seqs_label = mcs_seqs_label.long()

        emb_list = []
        # Optional feature encodings
        if self.include_num_rbs_cond:
            num_rbs_cond_enc = self.layer_num_rbs_cond_emb(num_rbs_seqs_label)
            emb_list.append(num_rbs_cond_enc)
        else:
            num_rbs_cond_enc = 0

        if self.include_mcs_cond:
            mcs_cond_enc = self.layer_mcs_cond_emb(mcs_seqs_label)
            emb_list.append(mcs_cond_enc)   
        else:
            mcs_cond_enc = 0

        if self.concat_conditions == 'concat_all':
            # 3) Concatenate along the last dimension
            # shape -> [B, S, sum_of_emb_dims]
            enc_cond_output = torch.cat(emb_list, dim=-1)
            final_output = torch.cat([enc_out, enc_cond_output], dim=-1)
        elif self.concat_conditions == 'sum_conds':
            # add only non-zero embeddings
            if self.include_num_rbs_cond:
                enc_cond_output = num_rbs_cond_enc
                if self.include_mcs_cond:
                    enc_cond_output += mcs_cond_enc
            elif self.include_mcs_cond:
                enc_cond_output = mcs_cond_enc
                if self.include_num_rbs_cond:
                    enc_cond_output += num_rbs_cond_enc
            else:
                enc_cond_output = 0
            final_output = torch.cat([enc_out, enc_cond_output], dim=-1)
        elif self.concat_conditions == 'sum_all':
            final_output = enc_out
            if self.include_num_rbs_cond:
                final_output += num_rbs_cond_enc
            if self.include_mcs_cond:
                final_output += mcs_cond_enc
        else:
            raise ValueError(f"Invalid concatenation type: {self.concat_conditions}. Choose from ['concat_all', 'sum_conds', 'sum_all'].")
    
        return final_output

    def forward(self, num_rbs_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, time_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.
        Returns:
            tensor: hidden states at event times.
        """
        # only linear ones need unsqueeze
        # convert type_seqs to int type for embedding
        num_rbs_seqs = num_rbs_seqs.long()
        mcs_seqs = mcs_seqs.long()
        mretx_seqs = mretx_seqs.long()
        rfailed_seqs = rfailed_seqs.long()

        # [batch_size, seq_len, hidden_size (d_model)]
        # Temporal and type encoding
        time_enc = self.layer_temporal_encoding(time_seqs)
        mretx_enc = self.layer_mretx_emb(mretx_seqs)

        # 2) Build a list to concatenate later (maybe)
        emb_list = [time_enc, mretx_enc]

        # Optional feature encodings
        if self.include_num_rbs:
            num_rbs_enc = self.layer_num_rbs_emb(num_rbs_seqs)
            emb_list.append(num_rbs_enc)
        else:
            num_rbs_enc = 0
        
        if self.include_mcs:
            mcs_enc = self.layer_mcs_emb(mcs_seqs)
            emb_list.append(mcs_enc)
        else:
            mcs_enc = 0

        if self.include_rfailed:
            rfailed_enc = self.layer_rfailed_emb(rfailed_seqs)
            emb_list.append(rfailed_enc)
        else:
            rfailed_enc = 0
        
        if self.concat_embeddings:
            # 3) Concatenate along the last dimension
            # shape -> [B, S, sum_of_emb_dims]
            enc_output = torch.cat(emb_list, dim=-1)

            # [batch_size, seq_len, hidden_size]
            for enc_layer in self.stack_layers:
                enc_output = enc_layer(
                    enc_output,
                    mask=attention_mask
                )
        else:
            # add only non-zero embeddings
            enc_output = mretx_enc
            if self.include_num_rbs:
                enc_output += num_rbs_enc
            if self.include_mcs:
                enc_output += mcs_enc
            if self.include_rfailed:
                enc_output += rfailed_enc

            # [batch_size, seq_len, hidden_size]
            for enc_layer in self.stack_layers:
                enc_output += time_enc
                enc_output = enc_layer(
                    enc_output,
                    mask=attention_mask
                )

        return enc_output
    

    def pool_enc(self, enc_out):
        if self.pooling == "last":
            # Select the last token's output
            enc_out = enc_out[:, -1:, :]  # [batch_size, 1, hidden_size]
        elif self.pooling == "mean":
            # Mean pooling
            enc_out = torch.mean(enc_out, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        elif self.pooling == "max":
            # Max pooling
            enc_out, _ = torch.max(enc_out, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        elif self.pooling == "min":
            # Min pooling
            enc_out, _ = torch.min(enc_out, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        elif self.pooling == "sum":
            # Sum pooling
            enc_out = torch.sum(enc_out, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling}. Choose from ['last', 'mean', 'max', 'min', 'sum'].")
        return enc_out

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        num_rbs_seqs = num_rbs_seqs[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs = dtime_seqs[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            num_rbs_seqs[:, :-1],
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1],
            attention_mask[:, :-1, :-1]
        )

        enc_out = self.pool_enc(enc_out)

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            enc_out,
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed= rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        mretx_ll = mretx_dist.log_prob(label_mretx) * event_mask
        rfailed_dist = Categorical(logits=rfailed_logits)
        rfailed_ll = rfailed_dist.log_prob(label_rfailed) * event_mask

        # [batch_size,]
        mretx_loss = -mretx_ll.sum()
        rfailed_loss = -rfailed_ll.sum()
        total_loss = mretx_loss + rfailed_loss

        num_events = event_mask.sum().item()
        return total_loss, num_events, mretx_loss, rfailed_loss


    def predict_mean_variance(self, batch, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        num_rbs_seqs = num_rbs_seqs[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs = dtime_seqs[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            num_rbs_seqs[:, :-1],
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1],
            attention_mask[:, :-1, :-1]
        )

        enc_out = self.pool_enc(enc_out)

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            enc_out,
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        rfailed_dist = Categorical(logits=rfailed_logits)

        # Calculate mean and variance manually for mretx
        mretx_indices = torch.arange(self.num_mretx_types, dtype=mretx_dist.probs.dtype, device=mretx_dist.probs.device)  # [0, 1, ..., n-1]
        pred_mretx = torch.sum(mretx_dist.probs * mretx_indices, dim=-1, keepdim=True)  # [batch_size, 1]
        pred_mretx_var = torch.sum(mretx_dist.probs * (mretx_indices ** 2), dim=-1, keepdim=True) - pred_mretx ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        # Calculate mean and variance manually for rfailed
        rfailed_indices = torch.arange(self.num_rfailed_types, dtype=rfailed_dist.probs.dtype, device=rfailed_dist.probs.device)  # [0, 1, ..., n-1]
        pred_rfailed = torch.sum(rfailed_dist.probs * rfailed_indices, dim=-1, keepdim=True)  # [batch_size, 1]
        pred_rfailed_var = torch.sum(rfailed_dist.probs * (rfailed_indices ** 2), dim=-1, keepdim=True) - pred_rfailed ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        num_events = event_mask.sum().item()
        return (pred_mretx[...,0,0],pred_mretx_var[...,0,0]), (pred_rfailed[...,0,0],pred_rfailed_var[...,0,0]), (label_mretx[...,0], label_rfailed[...,0]), event_mask[...,0], num_events


    def predict_probabilities(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        # only consider the last self.his_len events in the history
        num_rbs_seqs = num_rbs_seqs[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs = dtime_seqs[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            num_rbs_seqs[:, :-1],
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1],
            attention_mask[:, :-1, :-1]
        )

        enc_out = self.pool_enc(enc_out)

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            enc_out,
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]
        mretx_logprobs = torch.log_softmax(mretx_logits, dim=-1)
        rfailed_logprobs = torch.log_softmax(rfailed_logits, dim=-1)

        num_events = event_mask.sum().item()

        return (mretx_logprobs, rfailed_logprobs), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events

        
    def generate_samples(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

                # only consider the last self.his_len events in the history
        num_rbs_seqs = num_rbs_seqs[:, -1 -self.his_len:]
        mcs_seqs = mcs_seqs[:, -1 -self.his_len:]
        mretx_seqs = mretx_seqs[:, -1 -self.his_len:]
        rfailed_seqs = rfailed_seqs[:, -1 -self.his_len:]
        time_seqs = time_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs = dtime_seqs[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            num_rbs_seqs[:, :-1],
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1],
            attention_mask[:, :-1, :-1]
        )

        enc_out = self.pool_enc(enc_out)

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            enc_out,
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        rfailed_dist = Categorical(logits=rfailed_logits)

        mretx_samples = mretx_dist.sample((prediction_config['num_samples_mretx'],))
        rfailed_samples = rfailed_dist.sample((prediction_config['num_samples_rfailed'],))

        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        return (mretx_samples, rfailed_samples), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events
    

class SingleStepRETXPriorCond(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(SingleStepRETXPriorCond, self).__init__(model_config)

        self.concat_conditions = model_config.model_specs['conditions']['concat']

        self.include_num_rbs_cond = model_config.model_specs['conditions']['include_num_rbs']
        if self.concat_conditions == 'concat_all':
            self.num_rbs_cond_emb_dim = model_config.model_specs['conditions']['num_rbs_emb_dim']
        elif self.concat_conditions == 'sum_conds' or self.concat_conditions == 'sum_all':
            self.conds_emb_dim = model_config.model_specs['conditions']['cond_dim']
            self.num_rbs_cond_emb_dim = self.conds_emb_dim
        else:
            raise ValueError(f"Invalid concatenation type: {self.concat_conditions}. Choose from ['concat_all', 'sum_conds', 'sum_all'].")

        self.include_mcs_cond = model_config.model_specs['conditions']['include_mcs']
        if self.concat_conditions == 'concat_all':
            self.mcs_cond_emb_dim = model_config.model_specs['conditions']['mcs_emb_dim']
        elif self.concat_conditions == 'sum_conds' or self.concat_conditions == 'sum_all':
            self.conds_emb_dim = model_config.model_specs['conditions']['cond_dim']
            self.mcs_cond_emb_dim = self.conds_emb_dim
        else:
            raise ValueError(f"Invalid concatenation type: {self.concat_conditions}. Choose from ['concat_all', 'sum_conds', 'sum_all'].")


        if self.concat_conditions == 'concat_all':
            self.cond_hidden_size = self.num_rbs_cond_emb_dim + self.mcs_cond_emb_dim
        elif self.concat_conditions == 'sum_conds' or self.concat_conditions == 'sum_all':
            self.cond_hidden_size = self.conds_emb_dim
        else:
            raise ValueError(f"Invalid concatenation type: {self.concat_conditions}. Choose from ['concat_all', 'sum_conds', 'sum_all'].")
        

        # size of transformer tokens stays fixed
        self.d_model = self.cond_hidden_size

        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate
        
        # History embedding configurations
        # num rbs embedding
        self.num_rbs_types = 107  # slot indices: 0 to 19 (20 types), and padding token
        self.num_rbs_pad_id = 106
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
        if self.include_num_rbs_cond:
            # slot number encoding
            self.layer_num_rbs_cond_emb = nn.Embedding(
                self.num_rbs_types,
                self.num_rbs_cond_emb_dim,
                padding_idx=self.num_rbs_pad_id,
                device=self.device
            )
        if self.include_mcs_cond:
            # mcs encoding
            self.layer_mcs_cond_emb = nn.Embedding(
                self.num_mcs_types,
                self.mcs_cond_emb_dim,
                padding_idx=self.mcs_pad_id,
                device=self.device
            )

        # MLP layer (self.feed_forward)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # mretx prediction linear layer
        self.mretx_linear = nn.Linear(self.d_model, self.num_mretx_types)
        # rfailed prediction linear layer
        self.rfailed_linear = nn.Linear(self.d_model, self.num_rfailed_types)


    def forward_cond(self, num_rbs_seqs_label, mcs_seqs_label):
        num_rbs_seqs_label = num_rbs_seqs_label.long()
        mcs_seqs_label = mcs_seqs_label.long()

        emb_list = []
        # Optional feature encodings
        if self.include_num_rbs_cond:
            num_rbs_cond_enc = self.layer_num_rbs_cond_emb(num_rbs_seqs_label)
            emb_list.append(num_rbs_cond_enc)
        else:
            num_rbs_cond_enc = 0

        if self.include_mcs_cond:
            mcs_cond_enc = self.layer_mcs_cond_emb(mcs_seqs_label)
            emb_list.append(mcs_cond_enc)   
        else:
            mcs_cond_enc = 0

        if self.concat_conditions == 'concat_all':
            # 3) Concatenate along the last dimension
            # shape -> [B, S, sum_of_emb_dims]
            final_output = torch.cat(emb_list, dim=-1)
        elif self.concat_conditions == 'sum_conds' or self.concat_conditions == 'sum_all':
            # add only non-zero embeddings
            if self.include_num_rbs_cond:
                enc_cond_output = num_rbs_cond_enc
                if self.include_mcs_cond:
                    enc_cond_output += mcs_cond_enc
            elif self.include_mcs_cond:
                enc_cond_output = mcs_cond_enc
                if self.include_num_rbs_cond:
                    enc_cond_output += num_rbs_cond_enc
            else:
                enc_cond_output = 0
            final_output = enc_cond_output
        else:
            raise ValueError(f"Invalid concatenation type: {self.concat_conditions}. Choose from ['concat_all', 'sum_conds', 'sum_all'].")
    
        return final_output

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed= rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        mretx_ll = mretx_dist.log_prob(label_mretx) * event_mask
        rfailed_dist = Categorical(logits=rfailed_logits)
        rfailed_ll = rfailed_dist.log_prob(label_rfailed) * event_mask

        # [batch_size,]
        mretx_loss = -mretx_ll.sum()
        rfailed_loss = -rfailed_ll.sum()
        total_loss = mretx_loss + rfailed_loss

        num_events = event_mask.sum().item()
        return total_loss, num_events, mretx_loss, rfailed_loss


    def predict_mean_variance(self, batch, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        rfailed_dist = Categorical(logits=rfailed_logits)

        # Calculate mean and variance manually for mretx
        mretx_indices = torch.arange(self.num_mretx_types, dtype=mretx_dist.probs.dtype, device=mretx_dist.probs.device)  # [0, 1, ..., n-1]
        pred_mretx = torch.sum(mretx_dist.probs * mretx_indices, dim=-1, keepdim=True)  # [batch_size, 1]
        pred_mretx_var = torch.sum(mretx_dist.probs * (mretx_indices ** 2), dim=-1, keepdim=True) - pred_mretx ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        # Calculate mean and variance manually for rfailed
        rfailed_indices = torch.arange(self.num_rfailed_types, dtype=rfailed_dist.probs.dtype, device=rfailed_dist.probs.device)  # [0, 1, ..., n-1]
        pred_rfailed = torch.sum(rfailed_dist.probs * rfailed_indices, dim=-1, keepdim=True)  # [batch_size, 1]
        pred_rfailed_var = torch.sum(rfailed_dist.probs * (rfailed_indices ** 2), dim=-1, keepdim=True) - pred_rfailed ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        num_events = event_mask.sum().item()
        return (pred_mretx[...,0,0],pred_mretx_var[...,0,0]), (pred_rfailed[...,0,0],pred_rfailed_var[...,0,0]), (label_mretx[...,0], label_rfailed[...,0]), event_mask[...,0], num_events


    def predict_probabilities(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]

        # [batch_size, seq_len]
        mretx_logprobs = torch.log_softmax(mretx_logits, dim=-1)
        rfailed_logprobs = torch.log_softmax(rfailed_logits, dim=-1)
        num_events = event_mask.sum().item()

        return (mretx_logprobs, rfailed_logprobs), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events

        
    def generate_samples(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        # input: [batch_size, 1, hidden_size]
        enc_cond_out = self.forward_cond(
            num_rbs_seqs[:, -1:], #label num_rbs (condition)
            mcs_seqs[:, -1:], # label mcs (condition) 
        )
        # output: [batch_size, 1, cond_hidden_size]

        # [batch_size, seq_len, num_mcs_types]
        mretx_logits = self.mretx_linear(enc_cond_out)
        rfailed_logits = self.rfailed_linear(enc_cond_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        rfailed_dist = Categorical(logits=rfailed_logits)

        mretx_samples = mretx_dist.sample((prediction_config['num_samples_mretx'],))
        rfailed_samples = rfailed_dist.sample((prediction_config['num_samples_rfailed'],))

        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        return (mretx_samples, rfailed_samples), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events
    

class SingleStepRETXPrior(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(SingleStepRETXPrior, self).__init__(model_config)

        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate
        
        # retransmissions embedding
        self.num_mretx_types = 5  # retransmission indices: 0 to 3 (4 types), and padding token
        self.mretx_pad_id = 4
        # rlc failed embedding
        self.num_rfailed_types = 3  # failed attempt indices: 0 and 1 (2 types), and padding token
        self.rfailed_pad_id = 2
        
        # RETX prediction linear layer
        self.mretx_linear = nn.Parameter(torch.empty(self.num_mretx_types, device=self.device))
        nn.init.uniform_(self.mretx_linear, a=0.0, b=1.0)

        # rfailed prediction linear layer
        self.rfailed_linear = nn.Parameter(torch.empty(self.num_rfailed_types, device=self.device))
        nn.init.uniform_(self.rfailed_linear, a=0.0, b=1.0)


    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        batch_size, seq_len = mcs_seqs[:, :-1].shape
        # Unsqueeze to add batch and sequence dimensions
        # Shape: [1, 1, num_marks]
        expanded_mretx_linear = self.mretx_linear.unsqueeze(0).unsqueeze(0)  
        expanded_rfailed_linear = self.rfailed_linear.unsqueeze(0).unsqueeze(0) 

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mretx_logits = expanded_mretx_linear.repeat(batch_size, seq_len, 1)
        rfailed_logits = expanded_rfailed_linear.repeat(batch_size, seq_len, 1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed= rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        mretx_ll = mretx_dist.log_prob(label_mretx) * event_mask
        rfailed_dist = Categorical(logits=rfailed_logits)
        rfailed_ll = rfailed_dist.log_prob(label_rfailed) * event_mask

        # [batch_size,]
        mretx_loss = -mretx_ll.sum()
        rfailed_loss = -rfailed_ll.sum()
        total_loss = mretx_loss + rfailed_loss

        num_events = event_mask.sum().item()
        return total_loss, num_events, mretx_loss, rfailed_loss


    def predict_mean_variance(self, batch, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        batch_size, seq_len = mcs_seqs[:, :-1].shape
        # Unsqueeze to add batch and sequence dimensions
        # Shape: [1, 1, num_marks]
        expanded_mretx_linear = self.mretx_linear.unsqueeze(0).unsqueeze(0)  
        expanded_rfailed_linear = self.rfailed_linear.unsqueeze(0).unsqueeze(0) 

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mretx_logits = expanded_mretx_linear.repeat(batch_size, seq_len, 1)
        rfailed_logits = expanded_rfailed_linear.repeat(batch_size, seq_len, 1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        rfailed_dist = Categorical(logits=rfailed_logits)

        # Calculate mean and variance manually for mretx
        mretx_indices = torch.arange(self.num_mretx_types, dtype=mretx_dist.probs.dtype, device=mretx_dist.probs.device)  # [0, 1, ..., n-1]
        pred_mretx = torch.sum(mretx_dist.probs * mretx_indices, dim=-1, keepdim=True)  # [batch_size, 1]
        pred_mretx_var = torch.sum(mretx_dist.probs * (mretx_indices ** 2), dim=-1, keepdim=True) - pred_mretx ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        # Calculate mean and variance manually for rfailed
        rfailed_indices = torch.arange(self.num_rfailed_types, dtype=rfailed_dist.probs.dtype, device=rfailed_dist.probs.device)  # [0, 1, ..., n-1]
        pred_rfailed = torch.sum(rfailed_dist.probs * rfailed_indices, dim=-1, keepdim=True)  # [batch_size, 1]
        pred_rfailed_var = torch.sum(rfailed_dist.probs * (rfailed_indices ** 2), dim=-1, keepdim=True) - pred_rfailed ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        num_events = event_mask.sum().item()
        return (pred_mretx[...,0,0],pred_mretx_var[...,0,0]), (pred_rfailed[...,0,0],pred_rfailed_var[...,0,0]), (label_mretx[...,0], label_rfailed[...,0]), event_mask[...,0], num_events


    def predict_probabilities(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        batch_size, seq_len = mcs_seqs[:, :-1].shape
        # Unsqueeze to add batch and sequence dimensions
        # Shape: [1, 1, num_marks]
        expanded_mretx_linear = self.mretx_linear.unsqueeze(0).unsqueeze(0)  
        expanded_rfailed_linear = self.rfailed_linear.unsqueeze(0).unsqueeze(0) 

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mretx_logits = expanded_mretx_linear.repeat(batch_size, seq_len, 1)
        rfailed_logits = expanded_rfailed_linear.repeat(batch_size, seq_len, 1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]

        # [batch_size, seq_len]
        mretx_logrobs = torch.log_softmax(mretx_logits, dim=-1)
        rfailed_logprobs = torch.log_softmax(rfailed_logits, dim=-1)
        num_events = event_mask.sum().item()

        return (mretx_logrobs, rfailed_logprobs), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events

        
    def generate_samples(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seqs, dtime_seqs, type_seqs, mcs_seqs, num_rbs_seqs, mretx_seqs, rfailed_seqs, batch_non_pad_mask, attention_mask = batch

        batch_size, seq_len = mcs_seqs[:, :-1].shape
        # Unsqueeze to add batch and sequence dimensions
        # Shape: [1, 1, num_marks]
        expanded_mretx_linear = self.mretx_linear.unsqueeze(0).unsqueeze(0)  
        expanded_rfailed_linear = self.rfailed_linear.unsqueeze(0).unsqueeze(0) 

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mretx_logits = expanded_mretx_linear.repeat(batch_size, seq_len, 1)
        rfailed_logits = expanded_rfailed_linear.repeat(batch_size, seq_len, 1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mretx = mretx_seqs[:, -1:]
        label_rfailed = rfailed_seqs[:, -1:]
        mretx_dist = Categorical(logits=mretx_logits)
        rfailed_dist = Categorical(logits=rfailed_logits)

        mretx_samples = mretx_dist.sample((prediction_config['num_samples_mretx'],))
        rfailed_samples = rfailed_dist.sample((prediction_config['num_samples_rfailed'],))

        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        return (mretx_samples, rfailed_samples), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events