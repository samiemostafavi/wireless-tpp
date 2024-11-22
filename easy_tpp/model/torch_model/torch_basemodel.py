""" Base model with common functionality  """

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D

from easy_tpp.model.torch_model.torch_thinning import EventSampler
from easy_tpp.utils import set_device


class TorchBaseModel(nn.Module):
    def __init__(self, model_config):
        """Initialize the BaseModel

        Args:
            model_config (EasyTPP.ModelConfig): model spec of configs
        """
        super(TorchBaseModel, self).__init__()
        self.loss_integral_num_sample_per_step = model_config.loss_integral_num_sample_per_step
        self.hidden_size = model_config.hidden_size
        self.num_event_types = model_config.num_event_types  # not include [PAD], [BOS], [EOS]
        self.num_event_types_pad = model_config.num_event_types_pad  # include [PAD], [BOS], [EOS]
        self.pad_token_id = model_config.pad_token_id
        self.eps = torch.finfo(torch.float32).eps

        self.is_prior = model_config.model_specs.get('prior', False)

        if not self.is_prior and not (type(self).__name__ == 'IntensityFree2D'):
            self.layer_type_emb = nn.Embedding(self.num_event_types_pad,  # have padding
                                            self.hidden_size,
                                            padding_idx=self.pad_token_id)
        else:
            self.layer_type_emb = None

        if (type(self).__name__ == 'THP'):
            self.std_inter_time = model_config.get("std_inter_time", 1.0)
            self.transform = D.AffineTransform(loc=0, scale=self.std_inter_time)
        else:
            self.transform = None

        self.gen_config = model_config.thinning
        self.event_sampler = None
        self.device = set_device(model_config.gpu)
        self.use_mc_samples = model_config.use_mc_samples

        self.to(self.device)

        if self.gen_config:
            self.event_sampler = EventSampler(num_sample=self.gen_config.num_sample,
                                              num_exp=self.gen_config.num_exp,
                                              over_sample_rate=self.gen_config.over_sample_rate,
                                              patience_counter=self.gen_config.patience_counter,
                                              num_samples_boundary=self.gen_config.num_samples_boundary,
                                              dtime_max=self.gen_config.dtime_max,
                                              device=self.device,
                                              transform=self.transform)

    @staticmethod
    def generate_model_from_config(model_config):
        """Generate the model in derived class based on model config.

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        model_id = model_config.model_id

        for subclass in TorchBaseModel.__subclasses__():
            if subclass.__name__ == model_id:
                return subclass(model_config)

        raise RuntimeError('No model named ' + model_id)

    @staticmethod
    def get_logits_at_last_step(logits, batch_non_pad_mask, sample_len=None):
        """Retrieve the hidden states of last non-pad events.

        Args:
            logits (tensor): [batch_size, seq_len, hidden_dim], a sequence of logits
            batch_non_pad_mask (tensor): [batch_size, seq_len], a sequence of masks
            sample_len (tensor): default None, use batch_non_pad_mask to find out the last non-mask position

        ref: https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Returns:
            tensor: retrieve the logits of EOS event
        """

        seq_len = batch_non_pad_mask.sum(dim=1)
        select_index = seq_len - 1 if sample_len is None else seq_len - 1 - sample_len
        # [batch_size, hidden_dim]
        select_index = select_index.unsqueeze(1).repeat(1, logits.size(-1))
        # [batch_size, 1, hidden_dim]
        select_index = select_index.unsqueeze(1)
        # [batch_size, hidden_dim]
        last_logits = torch.gather(logits, dim=1, index=select_index).squeeze(1)
        return last_logits

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

    def make_dtime_loss_samples(self, time_delta_seq_transformed):
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
        sampled_dtimes_transformed = time_delta_seq_transformed[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes_transformed

    def compute_states_at_sample_times(self, **kwargs):
        raise NotImplementedError('This need to implemented in inherited class ! ')

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

            #if (type(self).__name__ == 'THP'):
            #    dtimes_pred = self.transform(dtimes_pred)

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

        return time_delta_seq[:, -num_step:], event_seq[:, -num_step:], \
               time_delta_seq_label, event_seq_label


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

        if (type(self).__name__ == 'THP'):
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
        if (type(self).__name__ == 'THP'):
            pdf_t = pdf_t / self.transform.scale
        dtimes_logprob = torch.log(pdf_t)

        # Compute probabilities for each event type
        lambda_k_t = intensities[:,1:,:]  # [batch_size, num_steps_dtime-1, event_num]
        lambda_t_total = lambda_k_t.sum(dim=-1, keepdim=True) + self.eps  # [batch_size, num_steps_dtime-1, 1]
        types_probs_pred = torch.log(lambda_k_t / lambda_t_total)  # [batch_size, num_steps_dtime-1, event_num]

        return (dtimes_logprob, types_probs_pred) , time_delta_seq_label, time_seq_label, event_seq_label
    