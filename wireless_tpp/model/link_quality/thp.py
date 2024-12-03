import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F

from wireless_tpp.model.baselayer import EncoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus
from wireless_tpp.model.basemodel import TorchBaseModel
from wireless_tpp.model.link_quality.thinning import EventSampler

class THPLinkQuality(TorchBaseModel):
    """Torch implementation of Transformer Hawkes Process, ICML 2020, https://arxiv.org/abs/2002.09291.
    Note: Part of the code is collected from https://github.com/yangalan123/anhp-andtt/tree/master/thp.
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(THPLinkQuality, self).__init__(model_config)

        self.std_inter_time = model_config.get("std_inter_time", 1.0)
        self.transform = D.AffineTransform(loc=0, scale=self.std_inter_time)

        self.layer_type_emb = nn.Embedding(self.num_event_types_pad,  # have padding
                                            self.hidden_size,
                                            padding_idx=self.pad_token_id)

        if self.gen_config:
            self.event_sampler = EventSampler(num_sample=self.gen_config.num_sample,
                                                num_exp=self.gen_config.num_exp,
                                                over_sample_rate=self.gen_config.over_sample_rate,
                                                patience_counter=self.gen_config.patience_counter,
                                                num_samples_boundary=self.gen_config.num_samples_boundary,
                                                dtime_max=self.gen_config.dtime_max,
                                                device=self.device,
                                                transform=self.transform)

        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        self.filter_mcs_events_for_loss = model_config.model_specs.get("filter_mcs_events_for_loss", False)
        if self.filter_mcs_events_for_loss:
            self.num_event_types_no_mcs_t = torch.tensor(model_config.num_event_types_no_mcs, device=self.device)

        # done in Base Model
        #self.std_inter_time = model_config.get("std_inter_time", 1.0)
        #self.transform = D.AffineTransform(loc=0, scale=self.std_inter_time)

        self.layer_temporal_encoding = TimePositionalEncoding(self.d_model, device=self.device)

        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)   # learnable mark-specific beta

        # Add MLP layer
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

    def forward(self, time_seqs, type_seqs, attention_mask):
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

        # [batch_size, seq_len, hidden_size]
        tem_enc = self.layer_temporal_encoding(time_seqs)
        enc_output = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        for enc_layer in self.stack_layers:
            enc_output += tem_enc
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask)

        return enc_output

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        time_seqs, time_delta_seqs_transformed, type_seqs, batch_non_pad_mask, attention_mask = batch

        type_seqs = type_seqs.long()

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])

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

        if self.filter_mcs_events_for_loss:
            # Create an event type mask to exclude mcs event types
            event_type_mask = (torch.arange(self.num_event_types, device=self.device) < self.num_event_types_no_mcs_t).float()
            event_type_mask = event_type_mask.view(1, 1, -1)  # Shape: [1, 1, num_event_types]
            # Apply the mcs event mask to lambda_at_event to zero out mcs event types
            lambda_at_event = lambda_at_event * event_type_mask  # Exclude mcs event types
            # For mcs events, set the intensity at the event type to 1.0 (so log(1) = 0)
            # Get indices where mcs events occur
            # Identify mcs events and non-mcs events
            is_mcs_event = (type_seqs[:, 1:] >= self.num_event_types_no_mcs_t)
            is_not_mcs_event = ~is_mcs_event  # Logical NOT to get non-mcs events
            mcs_event_indices = is_mcs_event

            batch_indices, time_indices = torch.nonzero(mcs_event_indices, as_tuple=True)
            event_types_mcs = type_seqs[:, 1:][batch_indices, time_indices]
            # Set lambda_at_event at these positions to 1.0
            lambda_at_event[batch_indices, time_indices, event_types_mcs] = 1.0

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


        if self.filter_mcs_events_for_loss:
            # Apply the event type mask to lambda_t_sample
            event_type_mask_expanded = event_type_mask.unsqueeze(2)  # Shape: [1, 1, 1, num_event_types]
            lambda_t_sample = lambda_t_sample * event_type_mask_expanded  # Exclude mcs event types
            # Adjust the sequence mask to exclude mcs events
            seq_mask = batch_non_pad_mask[:, 1:]
            seq_mask_no_mcs = seq_mask & is_not_mcs_event  # Exclude mcs events from the mask

            # Set mcs events to pad_token_id
            type_seqs = type_seqs.clone()  # Clone to avoid modifying the original tensor
            complete_is_mcs_event = (type_seqs >= self.num_event_types_no_mcs_t)
            type_seqs[complete_is_mcs_event] = self.pad_token_id

            event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                            lambdas_loss_samples=lambda_t_sample,
                                                                            time_delta_seq=time_delta_seqs[:, 1:],
                                                                            seq_mask=seq_mask_no_mcs,
                                                                            type_seq=type_seqs[:, 1:])
        else:
            event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                            lambdas_loss_samples=lambda_t_sample,
                                                                            time_delta_seq=time_delta_seqs[:, 1:],
                                                                            seq_mask=batch_non_pad_mask[:, 1:],
                                                                            type_seq=type_seqs[:, 1:])

        # compute loss to minimize
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events


    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else: # Use trapezoid rule
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes

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

    def predict_probabilities_one_step_since_last_event(self, batch, prediction_config, forward=False):
        """Single-step event probability prediction since last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        if self.includes_mcs:
            time_seq_label, time_delta_seq_label, event_seq_label, mcs_seq_label, batch_non_pad_mask_label, type_mask_label = batch

            if not forward:
                time_seq = time_seq_label[:, :-1]
                time_delta_seq = time_delta_seq_label[:, :-1]
                event_seq = event_seq_label[:, :-1]
                mcs_seq = mcs_seq_label[:, :-1]
            else:
                time_seq, time_delta_seq, event_seq, mcs_seq = time_seq_label, time_delta_seq_label, event_seq_label, mcs_seq_label

        else:
            time_seq_label, time_delta_seq_label, event_seq_label, batch_non_pad_mask_label, type_mask_label = batch
            
            if not forward:
                time_seq = time_seq_label[:, :-1]
                time_delta_seq = time_delta_seq_label[:, :-1]
                event_seq = event_seq_label[:, :-1]
            else:
                time_seq, time_delta_seq, event_seq = time_seq_label, time_delta_seq_label, event_seq_label
        

        #if type(self).__name__ == 'IntensityFree':
        #    probs_t = self.predict_prob_at_every_event(batch)
        #    return probs_t, time_delta_seq_label[:, -2:], event_seq_label[:, -2:]
        #else:

        # Expand dimensions to match batch and sequence sizes
        # time_since_last_event = time_since_last_event[None, None, :]
        batch_size, seq_len = time_seq.size()

        # Assume you have the time intervals you are interested in
        # For example, time_since_last_event is a tensor of times since the last event
        sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
        sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
        num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
        time_since_last_event = torch.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime, device=self.device)

        # Compute intensities at these times
        intensities = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            event_seq,
            time_since_last_event,
            max_steps=seq_len,
            compute_last_step_only=True
        )  # Shape: [batch_size, seq_len, num_steps_dtime, event_num]

        # seq_len is not relevant since we only look at the last event
        intensities = intensities[:,-1,:,:]  # Shape: [batch_size, num_steps_dtime, event_num]

        # Transform time_since_last_event to match the time scale of the model
        time_since_last_event = self.transform.inv(time_since_last_event)

        # Compute the cumulative intensity over time using cumulative trapezoidal integration
        delta_t = time_since_last_event[1:] - time_since_last_event[:-1]  # Time differences
        mid_intensities = (intensities[:, 1:, :] + intensities[:, :-1, :]) / 2  # Average intensities

        # Sum over event types to get total intensity
        total_intensity = mid_intensities.sum(dim=-1)  # Shape: [batch_size, num_steps_dtime - 1]

        # Compute cumulative integral
        cumulative_intensity = torch.cumsum(total_intensity * delta_t, dim=-1)  # [batch_size, num_steps_dtime - 1]

        # Compute survival function
        survival_function = torch.exp(-cumulative_intensity)  # [batch_size, num_steps_dtime - 1]

        # Adjust intensities and survival function to align shapes
        lambda_t = total_intensity  # [batch_size, num_steps_dtime - 1]
        S_t = survival_function  # [batch_size, num_steps_dtime - 1]

        # Compute PDF
        pdf_t = lambda_t * S_t  # [batch_size, num_steps_dtime - 1]

        # Normalize PDF
        pdf_t = pdf_t / self.transform.scale
        dtimes_logprob = torch.log(pdf_t)

        # Compute probabilities for each event type
        lambda_k_t = intensities[:,1:,:]  # [batch_size, num_steps_dtime-1, event_num]
        lambda_t_total = lambda_k_t.sum(dim=-1, keepdim=True) + self.eps  # [batch_size, num_steps_dtime-1, 1]
        types_probs_pred = torch.log(lambda_k_t / lambda_t_total)  # [batch_size, num_steps_dtime-1, event_num]

        return (dtimes_logprob, types_probs_pred) , time_delta_seq_label, time_seq_label, event_seq_label

    
    def predict_one_step_at_every_event(self, batch):
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        # [batch_size, seq_len]
        dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max,
                                   time_delta_seq + self.event_sampler.dtime_max)

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                                              time_delta_seq,
                                                                              event_seq,
                                                                              dtime_boundary,
                                                                              self.compute_intensities_at_sample_times,
                                                                              compute_last_step_only=False)  # make it explicit

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                        time_delta_seq,
                                                                        event_seq,
                                                                        accepted_dtimes)

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_weighted, dim=-1)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
        return dtimes_pred, types_pred

    def generate_samples_one_step_since_last_event(self, batch, forward=False):

        time_seq_label, time_delta_seq_label, event_seq_label, batch_non_pad_mask_label, type_mask_label = batch
        time_seq = time_seq_label[:, :-1]
        time_delta_seq = time_delta_seq_label[:, :-1]
        event_seq = event_seq_label[:, :-1]
        
        # [batch_size, seq_len]
        dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

        # [batch_size, 1, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                        time_delta_seq,
                                                        event_seq,
                                                        dtime_boundary,
                                                        self.compute_intensities_at_sample_times,
                                                        compute_last_step_only=True)

        # [batch_size, 1]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

        # [batch_size, seq_len, 1, event_num]
        intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                        time_delta_seq,
                                                                        event_seq,
                                                                        dtimes_pred[:, :, None],
                                                                        max_steps=event_seq.size()[1])

        # [batch_size, seq_len, event_num]
        intensities_at_times = intensities_at_times.squeeze(dim=-2)

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_at_times, dim=-1)

        # [batch_size, 1]
        types_pred_ = types_pred[:, -1:]
        dtimes_pred_ = dtimes_pred[:, -1:]

        return dtimes_pred_, types_pred_, time_seq_label, event_seq_label

    def predict_multi_step_since_last_event(self, batch, forward=False):
        """Multi-step prediction since last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].
            num_step (int): num of steps for prediction.

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq_label, time_delta_seq_label, event_seq_label, batch_non_pad_mask_label, type_mask_label = batch

        num_step = self.gen_config.num_step_gen

        if not forward:
            time_seq = time_seq_label[:, :-num_step]
            time_delta_seq = time_delta_seq_label[:, :-num_step]
            event_seq = event_seq_label[:, :-num_step]
        else:
            time_seq, time_delta_seq, event_seq = time_seq_label, time_delta_seq_label, event_seq_label

        for i in range(num_step):
            # [batch_size, seq_len]
            dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

            # [batch_size, 1, num_sample]
            accepted_dtimes, weights = \
                self.event_sampler.draw_next_time_one_step(time_seq,
                                                           time_delta_seq,
                                                           event_seq,
                                                           dtime_boundary,
                                                           self.compute_intensities_at_sample_times,
                                                           compute_last_step_only=True)

            # [batch_size, 1]
            dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

            # [batch_size, seq_len, 1, event_num]
            intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                            time_delta_seq,
                                                                            event_seq,
                                                                            dtimes_pred[:, :, None],
                                                                            max_steps=event_seq.size()[1])

            # [batch_size, seq_len, event_num]
            intensities_at_times = intensities_at_times.squeeze(dim=-2)

            # [batch_size, seq_len]
            types_pred = torch.argmax(intensities_at_times, dim=-1)

            # [batch_size, 1]
            types_pred_ = types_pred[:, -1:]
            dtimes_pred_ = dtimes_pred[:, -1:]
            time_pred_ = time_seq[:, -1:] + dtimes_pred_

            # concat to the prefix sequence
            time_seq = torch.cat([time_seq, time_pred_], dim=-1)
            time_delta_seq = torch.cat([time_delta_seq, dtimes_pred_], dim=-1)
            event_seq = torch.cat([event_seq, types_pred_], dim=-1)

        return time_delta_seq[:, -num_step - 1:], event_seq[:, -num_step - 1:], \
               time_delta_seq_label[:, -num_step - 1:], event_seq_label[:, -num_step - 1:]