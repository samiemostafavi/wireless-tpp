import torch
from torch.distributions import Categorical
from torch import nn

from wireless_tpp.model.baselayer import EncoderLayer, MultiHeadAttention, TimePositionalEncoding
from wireless_tpp.model.basemodel import TorchBaseModel


class SingleStepMCS(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(SingleStepMCS, self).__init__(model_config)

        self.concat_embeddings = model_config.model_specs['embeddings']['concat']

        # size of transformer tokens stays fixed
        self.d_model = model_config.hidden_size
        self.his_len = model_config.model_specs['history']['length']

        self.type_emb_dim = model_config.model_specs['embeddings']['type_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_num_rbs = model_config.model_specs['history']['include_num_rbs']
        self.num_rbs_emb_dim = model_config.model_specs['embeddings']['num_rbs_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_mcs = model_config.model_specs['history']['include_mcs']
        self.mcs_emb_dim = model_config.model_specs['embeddings']['mcs_emb_dim'] if self.concat_embeddings else self.d_model
        
        self.include_mretx = model_config.model_specs['history']['include_mretx']
        self.mretx_emb_dim = model_config.model_specs['embeddings']['mretx_emb_dim'] if self.concat_embeddings else self.d_model

        self.include_rfailed = model_config.model_specs['history']['include_rfailed']
        self.rfailed_emb_dim = model_config.model_specs['embeddings']['rfailed_emb_dim'] if self.concat_embeddings else self.d_model

        self.time_noise_std = model_config.model_specs['history']['time_noise_std']

        self.pooling = model_config.model_specs['pooling']['type']

        if self.concat_embeddings:
            # size of time embedding is self.d_model minues the total size of the other embeddings
            self.time_emb_size = self.d_model - (
                int(self.include_num_rbs)*self.num_rbs_emb_dim + \
                int(self.include_mcs)*self.mcs_emb_dim + \
                int(self.include_mretx)*self.mretx_emb_dim + \
                int(self.include_rfailed)*self.rfailed_emb_dim + \
                self.type_emb_dim
            )
        else:
            self.time_emb_size = self.d_model
        
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

        self.num_event_types_pad = 3

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

        # MLP layer (self.feed_forward) without condition
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        print(self.n_head, self.d_model, self.dropout, self.his_len)
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
        
        # MCS prediction linear layer
        self.linear = nn.Linear(self.hidden_size, self.num_mcs_types)


    def forward(self, num_rbs_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, time_seqs, type_seqs, attention_mask):
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
        type_seqs = type_seqs.long()
        num_rbs_seqs = num_rbs_seqs.long()
        mcs_seqs = mcs_seqs.long()
        mretx_seqs = mretx_seqs.long()
        rfailed_seqs = rfailed_seqs.long()

        # [batch_size, seq_len, hidden_size (d_model)]
        # Temporal and type encoding
        time_enc = self.layer_temporal_encoding(time_seqs)
        type_enc = self.layer_type_emb(type_seqs) # it is either packet arrival, first segment or segments later

        # 2) Build a list to concatenate later (maybe)
        emb_list = [time_enc, type_enc]

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
            enc_output = type_enc
            if self.include_num_rbs:
                enc_output += num_rbs_enc
            if self.include_mcs:
                enc_output += mcs_enc
            if self.include_mretx:
                enc_output += mretx_enc
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
        type_seqs = type_seqs[:, -1 -self.his_len:]
        attention_mask = attention_mask[:, -1 -self.his_len:, -1 -self.his_len:]
        dtime_seqs = dtime_seqs[:, -1 -self.his_len:]
        batch_non_pad_mask = batch_non_pad_mask[:, -1 -self.his_len:]

        # add noise to time_seqs to avoid overfitting
        if self.time_noise_std > 0:
            time_seqs[:, :-1] = time_seqs[:, :-1] + torch.normal(mean=0.0, std=self.time_noise_std, size=time_seqs[:, :-1].size(), device=self.device)

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            num_rbs_seqs[:, :-1],
            mcs_seqs[:, :-1], 
            mretx_seqs[:, :-1], 
            rfailed_seqs[:, :-1], 
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        type_seqs = type_seqs.long()

        enc_out = self.pool_enc(enc_out)

        # [batch_size, seq_len, num_mcs_types]
        mcs_logits = self.linear(enc_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mcs = mcs_seqs[:, -1:]
        mcs_dist = Categorical(logits=mcs_logits)
        mcs_ll = mcs_dist.log_prob(label_mcs) * event_mask

        # [batch_size,]
        loss = -mcs_ll.sum()

        num_events = event_mask.sum().item()
        return loss, num_events


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
        type_seqs = type_seqs[:, -1 -self.his_len:]
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
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        type_seqs = type_seqs.long()

        enc_out = self.pool_enc(enc_out)

        # [batch_size, seq_len, num_mcs_types]
        mcs_logits = self.linear(enc_out)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mcs = mcs_seqs[:, -1:]
        mcs_dist = Categorical(logits=mcs_logits)

        # Calculate mean and variance manually
        indices = torch.arange(self.num_mcs_types, dtype=mcs_dist.probs.dtype, device=mcs_dist.probs.device)  # [0, 1, ..., n-1]

        # Mean
        pred_mcs = torch.sum(mcs_dist.probs * indices, dim=-1, keepdim=True)  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        # Variance
        pred_var = torch.sum(mcs_dist.probs * (indices ** 2), dim=-1, keepdim=True) - pred_mcs ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        # prepare one to last mcs for evaluation
        one_to_last_mcs = mcs_seqs[:, -2:-1]

        num_events = event_mask.sum().item()
        return (pred_mcs[...,0,0],pred_var[...,0,0]), (None, None), (label_mcs[...,0], one_to_last_mcs[...,0]), event_mask[...,0], num_events


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
        type_seqs = type_seqs[:, -1 -self.his_len:]
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
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        type_seqs = type_seqs.long()

        enc_out = self.pool_enc(enc_out)

        # [batch_size, seq_len, num_mcs_types]
        mcs_logits = self.linear(enc_out)

        # [batch_size, seq_len]
        mcs_logprobs = torch.log_softmax(mcs_logits, dim=-1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        return (mcs_logprobs, None), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events

        
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
        type_seqs = type_seqs[:, -1 -self.his_len:]
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
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        type_seqs = type_seqs.long()

        enc_out = self.pool_enc(enc_out)

        # [batch_size, seq_len, num_mcs_types]
        mcs_logits = self.linear(enc_out)
        label_mcs = mcs_seqs[:, -1:]
        mcs_dist = Categorical(logits=mcs_logits)
        mcs_samples = mcs_dist.sample((prediction_config['num_samples_mcs'],))

        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        return (mcs_samples, None), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events
    

class SingleStepMCSPrior(TorchBaseModel):
    """Torch implementation of MDN Learning of Temporal Point Processes
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(SingleStepMCSPrior, self).__init__(model_config)

        self.use_norm = model_config.use_ln
        self.dropout = model_config.dropout_rate

        # mcs embedding
        self.num_mcs_types = 30  # MCS indices: 0 to 28 (29 types), and padding token
        self.mcs_pad_id = 29
        
        # MCS prediction linear layer
        self.linear = nn.Parameter(torch.empty(self.num_mcs_types, device=self.device))
        nn.init.uniform_(self.linear, a=0.0, b=1.0)
    

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
        expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mcs_logits = expanded_linear.repeat(batch_size, seq_len, 1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mcs = mcs_seqs[:, -1:]
        mcs_dist = Categorical(logits=mcs_logits)
        mcs_ll = mcs_dist.log_prob(label_mcs) * event_mask

        # [batch_size,]
        loss = -mcs_ll.sum()

        num_events = event_mask.sum().item()
        return loss, num_events


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
        expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mcs_logits = expanded_linear.repeat(batch_size, seq_len, 1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        label_mcs = mcs_seqs[:, -1:]
        mcs_dist = Categorical(logits=mcs_logits)

        # Calculate mean and variance manually
        indices = torch.arange(self.num_mcs_types, dtype=mcs_dist.probs.dtype, device=mcs_dist.probs.device)  # [0, 1, ..., n-1]

        # Mean
        pred_mcs = torch.sum(mcs_dist.probs * indices, dim=-1, keepdim=True)  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        # Variance
        pred_var = torch.sum(mcs_dist.probs * (indices ** 2), dim=-1, keepdim=True) - pred_mcs ** 2  # [batch_size, 1]
        # output shape here is [batch_size, 1, 1]

        num_events = event_mask.sum().item()
        return (pred_mcs[...,0,0],pred_var[...,0,0]), (None, None), (label_mcs[...,0], None), event_mask[...,0], num_events


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
        expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mcs_logits = expanded_linear.repeat(batch_size, seq_len, 1)

        # [batch_size, seq_len]
        mcs_logprobs = torch.log_softmax(mcs_logits, dim=-1)
        
        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        return (mcs_logprobs, None), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events

        
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
        expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

        # Repeat the tensor across batch and sequence dimensions
        # Shape: [batch_size, seq_len, num_marks]
        mcs_logits = expanded_linear.repeat(batch_size, seq_len, 1)

        mcs_dist = Categorical(logits=mcs_logits)
        mcs_samples = mcs_dist.sample((prediction_config['num_samples_mcs'],))

        event_mask = batch_non_pad_mask[:, -1:]
        num_events = event_mask.sum().item()

        return (mcs_samples, None), (dtime_seqs, time_seqs, type_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs), event_mask, num_events