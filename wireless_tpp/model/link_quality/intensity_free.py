import torch
import torch.distributions as D
from torch import nn
from torch.distributions import Categorical, TransformedDistribution
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions import Normal as TorchNormal

from wireless_tpp.model.basemodel import TorchBaseModel


def clamp_preserve_gradients(x, min_val, max_val):
    """Clamp the tensor while preserving gradients in the clamped region.

    Args:
        x (tensor): tensor to be clamped.
        min_val (float): minimum value.
        max_val (float): maximum value.
    """
    return x + (x.clamp(min_val, max_val) - x).detach()


class Normal(TorchNormal):
    """Normal distribution, redefined `log_cdf` and `log_survival_function` due to
    no numerically stable implementation of them is available for normal distribution.
    """

    def log_cdf(self, x):
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x):
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)


class MixtureSameFamily(TorchMixtureSameFamily):
    """Mixture (same-family) distribution, redefined `log_cdf` and `log_survival_function`.
    """

    def log_cdf(self, x):
        x = self._pad(x)
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x):
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)


class NormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    Args:
        locs (tensor): [batch_size, seq_len, num_mix_components].
        log_scales (tensor): [batch_size, seq_len, num_mix_components].
        log_weights (tensor): [batch_size, seq_len, num_mix_components].
        mean_log_inter_time (float): Average log-inter-event-time.
        std_log_inter_time (float): Std of log-inter-event-times.
    """

    def __init__(self, locs, log_scales, log_weights, mean_inter_time, std_inter_time, validate_args=None):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=torch.exp(log_scales))
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_inter_time == 0.0 and std_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_inter_time, scale=std_inter_time)]

        self.mean_inter_time = mean_inter_time
        self.std_inter_time = std_inter_time

        self.transforms = transforms
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        self.sign = int(sign)
        super().__init__(GMM, transforms, validate_args=validate_args)

    def log_cdf(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_cdf(x)
        else:
            return self.base_dist.log_survival_function(x)

    def log_survival_function(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_survival_function(x)
        else:
            return self.base_dist.log_cdf(x)


class IntensityFreeLinkQuality(TorchBaseModel):
    """Torch implementation of Intensity-Free Learning of Temporal Point Processes, ICLR 2020.
    https://openreview.net/pdf?id=HygOjhEYDH

    reference: https://github.com/shchur/ifl-tpp
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(IntensityFreeLinkQuality, self).__init__(model_config)

        self.layer_type_emb = nn.Embedding(self.num_event_types_pad,  # have padding
                                            self.hidden_size,
                                            padding_idx=self.pad_token_id)

        self.num_mix_components = model_config.model_specs['num_mix_components']
        self.mean_inter_time = model_config.get("mean_inter_time", 0.0)
        self.std_inter_time = model_config.get("std_inter_time", 1.0)

        self.num_features = 2 + self.hidden_size # important 2 is for time and mcs

        if not self.is_prior:
            self.layer_rnn = nn.GRU(input_size=self.num_features,
                                    hidden_size=self.hidden_size,
                                    num_layers=2,  # used in original paper
                                    batch_first=True)
            
            self.mark_linear = nn.Linear(self.hidden_size, self.num_event_types_pad)
            self.linear = nn.Linear(self.hidden_size, 3 * self.num_mix_components)

        else:
            self.mark_linear = nn.Parameter(torch.empty(self.num_event_types_pad, device=self.device))
            self.linear = nn.Parameter(torch.empty( 3 * self.num_mix_components, device=self.device))
            nn.init.uniform_(self.mark_linear, a=0.0, b=1.0)
            nn.init.uniform_(self.linear, a=0.0, b=1.0)
        

        if self.mean_inter_time == 0.0 and self.std_inter_time == 1.0:
            self.transform = None
        else:
            self.transform = D.AffineTransform(loc=self.mean_inter_time, scale=self.std_inter_time)

    def forward(self, mcs_seqs, time_delta_seqs, type_seqs):
        """Call the model.

        Args:
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens.
        """
        # [batch_size, seq_len, hidden_size]
        # We dont normalize inter-event time here
        temporal_seqs = time_delta_seqs.unsqueeze(-1)

        mcs_seqs = mcs_seqs.unsqueeze(-1)

        # [batch_size, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 2], important 2 is for time and mcs
        rnn_input = torch.cat([mcs_seqs, temporal_seqs, type_emb], dim=-1)

        # [batch_size, seq_len, hidden_size]
        rnn_input = rnn_input.float()
        context = self.layer_rnn(rnn_input)[0]

        return context

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        mcs_seqs, time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        batch_size, seq_len = time_delta_seqs[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(mcs_seqs[:, :-1], time_delta_seqs[:, :-1], type_seqs[:, :-1])

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = self.linear(context)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 3 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 3 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = expanded_linear

            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, num_marks]
            expanded_mark_linear = self.mark_linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, num_marks]
            expanded_mark_linear = expanded_mark_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(expanded_mark_linear, dim=-1)
        
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0) # it was -5 to 3, but it was too small!
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time
        )

        #inter_times = time_delta_seqs[:, 1:].clamp(min=1e-5)
        inter_times = time_delta_seqs[:, 1:]
        # [batch_size, seq_len]
        event_mask = torch.logical_and(batch_non_pad_mask[:, 1:], type_seqs[:, 1:] != self.pad_token_id)
        time_ll = inter_time_dist.log_prob(inter_times) * event_mask

        mark_dist = Categorical(logits=mark_logits)
        mark_ll = mark_dist.log_prob(type_seqs[:, 1:]) * event_mask

        log_p = time_ll + mark_ll

        # [batch_size,]
        loss = -log_p.sum()

        num_events = event_mask.sum().item()
        return loss, num_events
    

    def predict_one_step_at_every_event(self, batch):
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        mcs_seq, time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch
        batch_size, seq_len = time_delta_seq[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(mcs_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1])

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = self.linear(context)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 3 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 3 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = expanded_linear

            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, num_marks]
            expanded_mark_linear = self.mark_linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, num_marks]
            expanded_mark_linear = expanded_mark_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(expanded_mark_linear, dim=-1)

        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time
        )

        # [num_samples, batch_size, seq_len]
        accepted_dtimes = inter_time_dist.sample((1000,))
        dtimes_pred = accepted_dtimes.mean(dim=0)

        # [batch_size, seq_len, num_marks]
        # Marks are modeled conditionally independently from times  
        types_pred = torch.argmax(mark_logits, dim=-1)
        return dtimes_pred, types_pred

    def predict_multi_step_since_last_event(self, batch, forward=False):
        """Multi-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
            tensor of loglikelihood loss, [seq_len].
        """
        mcs_seq, time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        batch_size, seq_len = time_delta_seqs[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(mcs_seq[:, :-1], time_delta_seqs[:, :-1], type_seqs[:, :-1])

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = self.linear(context)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 3 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 3 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = expanded_linear

            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, num_marks]
            expanded_mark_linear = self.mark_linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, num_marks]
            expanded_mark_linear = expanded_mark_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(expanded_mark_linear, dim=-1)
        
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0) # it was -5 to 3, but it was too small!
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time
        )

        #inter_times = time_delta_seqs[:, 1:].clamp(min=1e-5)
        inter_times = time_delta_seqs[:, 1:]
        # [batch_size, seq_len]
        event_mask = torch.logical_and(batch_non_pad_mask[:, 1:], type_seqs[:, 1:] != self.pad_token_id)
        time_ll = inter_time_dist.log_prob(inter_times) * event_mask

        mark_dist = Categorical(logits=mark_logits)
        mark_ll = mark_dist.log_prob(type_seqs[:, 1:]) * event_mask

        dtime_samples = inter_time_dist.sample((1000,))
        dtime_mean = dtime_samples.mean(dim=0)

        mark_samples = mark_dist.sample((1000,))
        mark_mean = mark_samples.float().mean(dim=0)

        num_events = event_mask.sum().item()
        return dtime_mean, mark_mean, time_ll.sum(), mark_ll.sum(), num_events
    


    def predict_probabilities_one_step_since_last_event(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        mcs_seq, time_seq, time_delta_seq, event_seq, _, _ = batch

        batch_size, seq_len = time_delta_seq[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(mcs_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1])

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = self.linear(context)

            # [batch_size, seq_len, num_marks]
            types_logprob_pred = torch.log_softmax(self.mark_linear(context), dim=-1)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 3 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 3 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = expanded_linear

            # Unsqueeze to add batch and sequence dimensionsß
            # Shape: [1, 1, num_marks]
            expanded_mark_linear = self.mark_linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, num_marks]
            expanded_mark_linear = expanded_mark_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, num_marks]
            types_logprob_pred = torch.log_softmax(expanded_mark_linear, dim=-1)

        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        # only select the last in seq_len
        locs, log_scales, log_weights = locs[:, -1:, :], log_scales[:, -1:, :], log_weights[:, -1:, :]

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 2.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time
        )

        sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
        sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
        num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
        time_since_last_event = torch.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime, device=self.device)
        dtimes_logprob_pred = inter_time_dist.log_prob(time_since_last_event)

        # [batch_size, seq_len, num_marks]
        # Marks are modeled conditionally independently from times
        types_logprob_pred = types_logprob_pred[:, -1, :]

        mcs_seq_label, time_seq_label, time_delta_seq_label, event_seq_label, _, _ = batch
        return dtimes_logprob_pred, types_logprob_pred, time_delta_seq_label, event_seq_label, mcs_seq_label
    

    def generate_samples_one_step_since_last_event(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        mcs_seq, time_seq, time_delta_seq, event_seq, _, _ = batch

        batch_size, seq_len = time_delta_seq[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(mcs_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1])

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = self.linear(context)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 3 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 3 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 3 * num_mix_components]
            raw_params = expanded_linear

            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, num_marks]
            expanded_mark_linear = self.mark_linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, num_marks]
            expanded_mark_linear = expanded_mark_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, num_marks]
            mark_logits = torch.log_softmax(expanded_mark_linear, dim=-1)

        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        # only select the last in seq_len
        locs, log_scales, log_weights = locs[:, -1:, :], log_scales[:, -1:, :], log_weights[:, -1:, :]

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 2.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = NormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time
        )

        dtimes_samples = inter_time_dist.sample((prediction_config['num_samples_dtime'],))

        event_type_dist = Categorical(logits=mark_logits)
        event_type_samples = event_type_dist.sample((prediction_config['num_samples_event_type'],))

        mcs_seq_label, time_seq_label, time_delta_seq_label, event_seq_label, _, _ = batch
        return (dtimes_samples, event_type_samples), time_delta_seq_label, event_seq_label