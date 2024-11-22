import torch
import torch.nn as nn
import torch.distributions as D

from easy_tpp.model.torch_model.torch_baselayer import EncoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class THPScheduling(TorchBaseModel):
    """Torch implementation of Transformer Hawkes Process, ICML 2020, https://arxiv.org/abs/2002.09291.
    Note: Part of the code is collected from https://github.com/yangalan123/anhp-andtt/tree/master/thp.
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(THPScheduling, self).__init__(model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        self.num_mcs_types_pad = 30  # MCS indices: 0 to 28 (29 types), and padding token
        self.mcs_pad_token_id = 29

        # parameter for filtering packet arrivals
        self.num_event_types_segment_only = self.num_event_types/2

        # temporal encoding
        self.layer_temporal_encoding = TimePositionalEncoding(self.d_model, device=self.device)

        # embedding for len_seqs, mcs_seqs, mac_retx_seqs, rlc_failed_seqs, num_rbs_seqs
        self.layer_len_emb = nn.Linear(1, self.d_model, device=self.device)
        self.layer_mcs_emb = nn.Embedding(self.num_mcs_types_pad,  # have padding
                                            self.d_model,
                                            padding_idx=self.mcs_pad_token_id,
                                            device=self.device)
        self.layer_mretx_emb = nn.Linear(1, self.d_model, device=self.device)
        self.layer_rlcf_emb = nn.Linear(1, self.d_model, device=self.device)
        self.layer_rbs_emb = nn.Linear(1, self.d_model, device=self.device)

        # Transformer layers (self.stack_layers) and MLP layer (self.feed_forward)
        # Equation (5)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        self.stack_layers = nn.ModuleList(
            [EncoderLayer(
                self.d_model,
                MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                   output_linear=False),
                use_residual=False,
                feed_forward=self.feed_forward,
                dropout=self.dropout
            ) for _ in range(self.n_layers)])

        # intensity parameters
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)   # learnable mark-specific beta


    def forward(self, len_seqs, mcs_seqs, mac_retx_seqs, rlc_failed_seqs, num_rbs_seqs, time_seqs, type_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        # convert type_seqs to int type for embedding
        type_seqs = type_seqs.long()
        mcs_seqs = mcs_seqs.long()
        len_seqs = len_seqs.float().unsqueeze(-1)
        mac_retx_seqs = mac_retx_seqs.float().unsqueeze(-1)
        rlc_failed_seqs = rlc_failed_seqs.float().unsqueeze(-1)
        num_rbs_seqs = num_rbs_seqs.float().unsqueeze(-1)

        # [batch_size, seq_len, hidden_size (d_model)]
        tem_enc = self.layer_temporal_encoding(time_seqs)
        type_enc = self.layer_type_emb(type_seqs)
        len_enc = self.layer_len_emb(len_seqs)
        mcs_enc = self.layer_mcs_emb(mcs_seqs)
        mretx_enc = self.layer_mretx_emb(mac_retx_seqs)
        rlcf_enc = self.layer_rlcf_emb(rlc_failed_seqs)
        rbs_enc = self.layer_rbs_emb(num_rbs_seqs)

        enc_output = type_enc + len_enc + mcs_enc + mretx_enc + rlcf_enc + rbs_enc

        # [batch_size, seq_len, hidden_size]
        for enc_layer in self.stack_layers:
            enc_output += tem_enc
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask
            )

        return enc_output

    def loglike_loss_old(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        len_seqs, mcs_seqs, mac_retx_seqs, rlc_failed_seqs, num_rbs_seqs, time_seqs, time_delta_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            len_seqs[:, :-1], 
            mcs_seqs[:, :-1], 
            mac_retx_seqs[:, :-1], 
            rlc_failed_seqs[:, :-1], 
            num_rbs_seqs[:, :-1],
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        type_seqs = type_seqs.long()
        mcs_seqs = mcs_seqs.long()

        # [batch_size, seq_len, num_event_types]
        # update time decay based on Equation (6)
        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # transform time_delta_seqs
        time_delta_seqs = self.transform.inv(time_delta_seqs_transformed)

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_event_types]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(
            enc_out) + factor_intensity_base
        
        lambda_at_event = self.softplus(intensity_states)

        # filter packet arrival events
        # Create an event type mask to exclude packet arrival event types
        event_type_mask = (torch.arange(self.num_event_types, device=self.device) < self.num_event_types_segment_only).float()
        event_type_mask = event_type_mask.view(1, 1, -1)  # Shape: [1, 1, num_event_types]
        # Apply the packet arrival mask to lambda_at_event to zero out packet arrival event types
        lambda_at_event = lambda_at_event * event_type_mask  # Exclude mcs event types
        # For packet arrival events, set the intensity at the event type to 1.0 (so log(1) = 0)
        # Get indices where packet arrival events occur
        # Identify packet arrival events and segment events
        is_packet_arrival_event = (type_seqs[:, 1:] >= self.num_event_types_segment_only)
        is_not_packet_arrival_event = ~is_packet_arrival_event  # Logical NOT to get non-mcs events
        packet_arrival_event_indices = is_packet_arrival_event

        batch_indices, time_indices = torch.nonzero(packet_arrival_event_indices, as_tuple=True)
        event_types_packet_arrival = type_seqs[:, 1:][batch_indices, time_indices]
        # Set lambda_at_event at these positions to 1.0
        lambda_at_event[batch_indices, time_indices, event_types_packet_arrival] = 1.0

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes_transformed = self.make_dtime_loss_samples(time_delta_seqs_transformed[:, 1:])

        # transform time_delta_seqs
        sample_dtimes = self.transform.inv(sample_dtimes_transformed)

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        # filter packet arrival events
        # Apply the event type mask to lambda_t_sample
        event_type_mask_expanded = event_type_mask.unsqueeze(2)  # Shape: [1, 1, 1, num_event_types]
        lambda_t_sample = lambda_t_sample * event_type_mask_expanded  # Exclude packet arrival event types
        # Adjust the sequence mask to exclude packet arrival events
        seq_mask = batch_non_pad_mask[:, 1:]
        seq_mask_no_packet_arrival = seq_mask & is_not_packet_arrival_event  # Exclude packet arrival events from the mask

        # Set packet arrival events to pad_token_id
        type_seqs = type_seqs.clone()  # Clone to avoid modifying the original tensor
        complete_is_packet_arrival_event = (type_seqs >= self.num_event_types_segment_only)
        type_seqs[complete_is_packet_arrival_event] = self.pad_token_id

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=seq_mask_no_packet_arrival,
                                                                        type_seq=type_seqs[:, 1:])


        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events
    

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        len_seqs, mcs_seqs, mac_retx_seqs, rlc_failed_seqs, num_rbs_seqs, time_seqs, time_delta_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(
            len_seqs[:, :-1], 
            mcs_seqs[:, :-1], 
            mac_retx_seqs[:, :-1], 
            rlc_failed_seqs[:, :-1], 
            num_rbs_seqs[:, :-1],
            time_seqs[:, :-1], 
            type_seqs[:, :-1], 
            attention_mask[:, :-1, :-1]
        )

        type_seqs = type_seqs.long()
        mcs_seqs = mcs_seqs.long()

        # [batch_size, seq_len, num_event_types]
        # update time decay based on Equation (6)
        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # transform time_delta_seqs
        time_delta_seqs = self.transform.inv(time_delta_seqs_transformed)

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_event_types]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(
            enc_out) + factor_intensity_base
        
        lambda_at_event = self.softplus(intensity_states)

        # filter all events other than type 0
        # Create an event type mask to exclude irrelevant events
        event_type_mask = (torch.arange(self.num_event_types, device=self.device) == 0).float()
        event_type_mask = event_type_mask.view(1, 1, -1)  # Shape: [1, 1, num_event_types]
        # Apply the mask to lambda_at_event to zero out irrelevant event types
        lambda_at_event = lambda_at_event * event_type_mask  # Exclude mcs event types
        # For irrelevant events, set the intensity at the event type to 1.0 (so log(1) = 0)
        # Get indices where irrelevant events occur
        # Identify irrelevant events and segment events
        is_irrelevant_event = (type_seqs[:, 1:] != 0)
        is_not_irrelevant_event = ~is_irrelevant_event  # Logical NOT to get non-mcs events
        irrelevant_event_indices = is_irrelevant_event

        batch_indices, time_indices = torch.nonzero(irrelevant_event_indices, as_tuple=True)
        event_types_irrelevant = type_seqs[:, 1:][batch_indices, time_indices]
        # Set lambda_at_event at these positions to 1.0
        lambda_at_event[batch_indices, time_indices, event_types_irrelevant] = 1.0

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes_transformed = self.make_dtime_loss_samples(time_delta_seqs_transformed[:, 1:])

        # transform time_delta_seqs
        sample_dtimes = self.transform.inv(sample_dtimes_transformed)

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        # filter packet arrival events
        # Apply the event type mask to lambda_t_sample
        event_type_mask_expanded = event_type_mask.unsqueeze(2)  # Shape: [1, 1, 1, num_event_types]
        lambda_t_sample = lambda_t_sample * event_type_mask_expanded  # Exclude packet arrival event types
        # Adjust the sequence mask to exclude packet arrival events
        seq_mask = batch_non_pad_mask[:, 1:]
        seq_mask_no_irrelevant = seq_mask & is_not_irrelevant_event  # Exclude packet arrival events from the mask

        # Set packet arrival events to pad_token_id
        type_seqs = type_seqs.clone()  # Clone to avoid modifying the original tensor
        complete_is_irrelevant_event = (type_seqs != 0)
        type_seqs[complete_is_irrelevant_event] = self.pad_token_id

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=seq_mask_no_irrelevant,
                                                                        type_seq=type_seqs[:, 1:])


        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events




    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(
            event_states) + factor_intensity_base

        return intensity_states

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes_transformed,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        attention_mask = kwargs.get('attention_mask', None)
        compute_last_step_only = kwargs.get('compute_last_step_only', False)
        batch_non_pad_mask = kwargs.get('batch_non_pad_mask', None)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_seqs, type_seqs, attention_mask)

        # transform time_delta_seqs
        sample_dtimes = self.transform.inv(sample_dtimes_transformed)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)

        if self.filter_mcs_events_for_loss:
            # Create an event type mask to exclude mcs event types
            event_type_mask = (torch.arange(self.num_event_types, device=self.device) < self.num_event_types_no_mcs_t).float()
            event_type_mask = event_type_mask.view(1, 1, -1)  # Shape: [1, 1, num_event_types]
            # Apply the event type mask to lambda
            event_type_mask_expanded = event_type_mask.unsqueeze(2)  # Shape: [1, 1, 1, num_event_types]
            lambdas = lambdas * event_type_mask_expanded  # Exclude mcs event types

        return lambdas
