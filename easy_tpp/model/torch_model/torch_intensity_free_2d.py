import torch
import torch.distributions as D
from torch import nn
from torch.distributions import TransformedDistribution
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions import MultivariateNormal as MultivariateTorchNormal
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.utils import logger


def clamp_preserve_gradients(x, min_val, max_val):
    """Clamp the tensor while preserving gradients in the clamped region.

    Args:
        x (tensor): tensor to be clamped.
        min_val (float): minimum value.
        max_val (float): maximum value.
    """
    return x + (x.clamp(min_val, max_val) - x).detach()


class MultivariateNormal(MultivariateTorchNormal):
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


class MultivariateNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of multivariate normal distributions.

    Args:
        locs (tensor): [batch_size, seq_len, num_mix_components].
        log_scales (tensor): [batch_size, seq_len, num_mix_components].
        log_weights (tensor): [batch_size, seq_len, num_mix_components].
        mean_log_inter_time (float): Average log-inter-event-time.
        std_log_inter_time (float): Std of log-inter-event-times.
    """

    def __init__(self, locs, log_scales, log_weights, mean_inter_time, std_inter_time, mean_event_type, std_event_type, device, validate_args=None):
        self.device = device

        mixture_dist = D.Categorical(logits=log_weights)

        # locs shape: [batch_size, seq_len, num_components, 2]
        # Reshape locs to have shape [num_components, 2] for 2D means
        num_components = locs.shape[-2]

        # Extract variances and covariance from log_scales
        # log_scales shape: [batch_size, seq_len, num_components, 3]
        variance_x = torch.exp(log_scales[..., 0])  # For x-axis
        variance_y = torch.exp(log_scales[..., 1])  # For y-axis
        covariance_xy = torch.tanh(log_scales[..., 2]) * (variance_x * variance_y).sqrt()  # Ensure valid range

        # Build scale_tril as a lower triangular matrix per component
        scale_tril = torch.zeros(log_scales.shape[0], log_scales.shape[1], num_components, 2, 2, device=self.device)
        scale_tril[..., 0, 0] = variance_x.sqrt()       # Variance of x
        scale_tril[..., 1, 1] = variance_y.sqrt()       # Variance of y
        scale_tril[..., 1, 0] = covariance_xy           # Covariance term

        component_dist = MultivariateNormal(loc=locs, scale_tril=scale_tril)
        GMM = MixtureSameFamily(mixture_dist, component_dist)

        if mean_inter_time == 0.0 and std_inter_time == 1.0 and mean_event_type == 0.0 and std_event_type == 1.0:
            transforms = []
        else:
            mean_2d = torch.tensor([mean_inter_time, mean_event_type], device=self.device)
            std_2d = torch.tensor([std_inter_time, std_event_type], device=self.device)
            transforms = [D.AffineTransform(loc=mean_2d, scale=std_2d)]

        self.mean_inter_time = mean_inter_time
        self.std_inter_time = std_inter_time
        self.mean_event_type = mean_event_type
        self.std_event_type = std_event_type

        self.transforms = transforms
        signX = 1
        for transform in self.transforms:
            signX = signX * transform.sign[0]
        self.signX = int(signX)    
        signY = 1
        for transform in self.transforms:
            signY = signY * transform.sign[1]
        self.signY = int(signY)
        super().__init__(GMM, transforms, validate_args=validate_args)

    def log_cdf(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.signX == 1:
            return self.base_dist.log_cdf(x)
        else:
            return self.base_dist.log_survival_function(x)

    def log_survival_function(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.signX == 1:
            return self.base_dist.log_survival_function(x)
        else:
            return self.base_dist.log_cdf(x)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cpu'):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class IntensityFree2D(TorchBaseModel):
    """Torch implementation of Intensity-Free Learning of Temporal Point Processes with continuous event type.
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(IntensityFree2D, self).__init__(model_config)

        self.num_mix_components = model_config.model_specs['num_mix_components']
        self.mean_inter_time = model_config.get("mean_inter_time", 0.0)
        self.std_inter_time = model_config.get("std_inter_time", 1.0)
        self.mean_event_type = model_config.get("mean_event_type", 0.0)
        self.std_event_type = model_config.get("std_event_type", 1.0)

        # Noise regularization, so far only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)
        if model_config.noise_regularization.event_type['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to event_type with std dev: {model_config.noise_regularization.event_type['std_dev']}")
            self.nr_event_type = AddGaussianNoise(mean=0, std=model_config.noise_regularization.event_type['std_dev'], device=self.device)
        else:
            self.nr_event_type = AddGaussianNoise(mean=0, std=0, device=self.device)

        #self.num_features = 1 + self.hidden_size
        #self.num_features = 2
        if not self.is_prior:
            self.layer_rnn = nn.GRU(input_size=2, # was self.num_features
                                    hidden_size=self.hidden_size,
                                    num_layers=model_config.get('num_layers',1),
                                    batch_first=True)
            self.linear = nn.Linear(self.hidden_size, 6 * self.num_mix_components)

        else:
            self.linear = nn.Parameter(torch.empty( 6 * self.num_mix_components, device=self.device))
            nn.init.uniform_(self.linear, a=0.0, b=1.0)
    

    def forward(self, time_delta_seqs, type_seqs):
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

        # [batch_size, seq_len, hidden_size]
        # We dont normalize types here, also we don't use type embedding
        temporal_types = type_seqs.unsqueeze(-1)

        # [batch_size, seq_len, hidden_size + 1]
        rnn_input = torch.cat([temporal_seqs, temporal_types], dim=-1)

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
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        batch_size, seq_len = time_delta_seqs[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(time_delta_seqs[:, :-1], type_seqs[:, :-1])

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = self.linear(context)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 6 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 6 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = expanded_linear
    

        # Extract locs, log_scales, and log_weights from raw_params
        # locs: Means in 2D, so [batch_size, seq_len, num_components, 2]
        locs = raw_params[..., :self.num_mix_components * 2].reshape(batch_size, seq_len, self.num_mix_components, 2)

        # log_scales: Diagonal and off-diagonal elements of the covariance matrix
        # [batch_size, seq_len, num_components, 3]
        # 3 values per component in 2D: [variance_x, variance_y, covariance_xy]
        log_scales = raw_params[..., self.num_mix_components * 2: self.num_mix_components * 2 + self.num_mix_components * 3].reshape(batch_size, seq_len, self.num_mix_components, 3)

        # log_weights: Unchanged, normalized weights for each component
        log_weights = raw_params[..., (self.num_mix_components * 2 + self.num_mix_components * 3):]
        log_weights = torch.log_softmax(log_weights, dim=-1)

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0) # it was -5 to 3, but it was too small!
        joint_dist = MultivariateNormalMixtureDistribution(
            locs=locs, # [batch_size, seq_len, num_components, 3]
            log_scales=log_scales, # [batch_size, seq_len, num_components, 3]
            log_weights=log_weights, # [batch_size, seq_len, num_components]
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time,
            mean_event_type=self.mean_event_type,
            std_event_type=self.std_event_type,
            device=self.device
        )

        inter_times = time_delta_seqs[:, 1:]
        event_types = type_seqs[:, 1:]
        # [batch_size, seq_len-1, 2]

        # apply noise regularization
        inter_times = self.nr_dtime(inter_times)
        event_types = self.nr_event_type(event_types)

        # stack and reshape to [batch_size, seq_len-1, 2]
        data = torch.stack((inter_times, event_types), dim=-1)
        joint_ll = joint_dist.log_prob(data)

        # joint_ll: [batch_size, seq_len-1]
        loss = -joint_ll.sum()

        # find number of events
        event_mask = torch.ones_like(type_seqs[:, 1:], dtype=torch.bool)
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
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # time_delta_seq should start from 1, because the first one is zero
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        batch_size, seq_len = time_delta_seq[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(time_delta_seq[:, :-1], event_seq[:, :-1])

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = self.linear(context)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 6 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 6 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = expanded_linear


        # Extract locs, log_scales, and log_weights from raw_params
        # locs: Means in 2D, so [batch_size, seq_len, num_components, 2]
        locs = raw_params[..., :self.num_mix_components * 2].reshape(batch_size, seq_len, self.num_mix_components, 2)

        # log_scales: Diagonal and off-diagonal elements of the covariance matrix
        # [batch_size, seq_len, num_components, 3]
        # 3 values per component in 2D: [variance_x, variance_y, covariance_xy]
        log_scales = raw_params[..., self.num_mix_components * 2: self.num_mix_components * 2 + self.num_mix_components * 3].reshape(batch_size, seq_len, self.num_mix_components, 3)

        # log_weights: Unchanged, normalized weights for each component
        log_weights = raw_params[..., (self.num_mix_components * 2 + self.num_mix_components * 3):]
        log_weights = torch.log_softmax(log_weights, dim=-1)

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0) # it was -5 to 3, but it was too small!
        joint_dist = MultivariateNormalMixtureDistribution(
            locs=locs, # [batch_size, seq_len, num_components, 3]
            log_scales=log_scales, # [batch_size, seq_len, num_components, 3]
            log_weights=log_weights, # [batch_size, seq_len, num_components]
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time,
            mean_event_type=self.mean_event_type,
            std_event_type=self.std_event_type,
            device=self.device
        )

        # [num_samples, batch_size, seq_len, 2]
        accepted_dtimes = joint_dist.sample((self.event_sampler.num_sample,))

        # [batch_size, seq_len, 2]
        joint_pred = accepted_dtimes.mean(dim=0)

        return joint_pred

    def predict_multi_step_since_last_event(self, batch, forward=False):
        """multi-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tensors of dtime and type prediction, [batch_size, seq_len].
            tensor of loglikelihood loss, [seq_len].
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # time_delta_seq should start from 1, because the first one is zero
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        batch_size, seq_len = time_delta_seq[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(time_delta_seq[:, :-1], event_seq[:, :-1])

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = self.linear(context)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 6 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 6 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = expanded_linear


        # Extract locs, log_scales, and log_weights from raw_params
        # locs: Means in 2D, so [batch_size, seq_len, num_components, 2]
        locs = raw_params[..., :self.num_mix_components * 2].reshape(batch_size, seq_len, self.num_mix_components, 2)

        # log_scales: Diagonal and off-diagonal elements of the covariance matrix
        # [batch_size, seq_len, num_components, 3]
        # 3 values per component in 2D: [variance_x, variance_y, covariance_xy]
        log_scales = raw_params[..., self.num_mix_components * 2: self.num_mix_components * 2 + self.num_mix_components * 3].reshape(batch_size, seq_len, self.num_mix_components, 3)

        # log_weights: Unchanged, normalized weights for each component
        log_weights = raw_params[..., (self.num_mix_components * 2 + self.num_mix_components * 3):]
        log_weights = torch.log_softmax(log_weights, dim=-1)

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0) # it was -5 to 3, but it was too small!
        joint_dist = MultivariateNormalMixtureDistribution(
            locs=locs, # [batch_size, seq_len, num_components, 3]
            log_scales=log_scales, # [batch_size, seq_len, num_components, 3]
            log_weights=log_weights, # [batch_size, seq_len, num_components]
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time,
            mean_event_type=self.mean_event_type,
            std_event_type=self.std_event_type,
            device=self.device
        )

        # [num_samples, batch_size, seq_len, 2]
        accepted_dtimes = joint_dist.sample((1000,))

        # [batch_size, seq_len, 2]
        joint_mean = accepted_dtimes.mean(dim=0)

        inter_times = time_delta_seq[:, 1:]
        event_types = event_seq[:, 1:]
        # [batch_size, seq_len-1, 2]

        # stack and reshape to [batch_size, seq_len-1, 2]
        data = torch.stack((inter_times, event_types), dim=-1)
        joint_ll = joint_dist.log_prob(data)

        # joint_ll: [batch_size, seq_len-1]
        # joint_ll: [seq_len-1]
        joint_ll = joint_ll.sum()

        # find number of events
        event_mask = torch.ones_like(event_seq[:, 1:], dtype=torch.bool)
        num_events = event_mask.sum().item()

        return joint_mean, torch.empty(size=joint_mean.shape,device=self.device), joint_ll, torch.empty(size=joint_ll.shape,device=self.device), num_events

    def predict_probabilities_one_step_since_last_event(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, _, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # time_delta_seq should start from 1, because the first one is zero
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        batch_size, seq_len = time_delta_seq[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(time_delta_seq[:, :-1], event_seq[:, :-1])

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = self.linear(context)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 6 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 6 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = expanded_linear

        # Extract locs, log_scales, and log_weights from raw_params
        # locs: Means in 2D, so [batch_size, seq_len, num_components, 2]
        locs = raw_params[..., :self.num_mix_components * 2].reshape(batch_size, seq_len, self.num_mix_components, 2)

        # log_scales: Diagonal and off-diagonal elements of the covariance matrix
        # [batch_size, seq_len, num_components, 3]
        # 3 values per component in 2D: [variance_x, variance_y, covariance_xy]
        log_scales = raw_params[..., self.num_mix_components * 2: self.num_mix_components * 2 + self.num_mix_components * 3].reshape(batch_size, seq_len, self.num_mix_components, 3)

        # log_weights: Unchanged, normalized weights for each component
        log_weights = raw_params[..., (self.num_mix_components * 2 + self.num_mix_components * 3):]
        log_weights = torch.log_softmax(log_weights, dim=-1)

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0) # it was -5 to 3, but it was too small!
        joint_dist = MultivariateNormalMixtureDistribution(
            locs=locs, # [batch_size, seq_len, num_components, 3]
            log_scales=log_scales, # [batch_size, seq_len, num_components, 3]
            log_weights=log_weights, # [batch_size, seq_len, num_components]
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time,
            mean_event_type=self.mean_event_type,
            std_event_type=self.std_event_type,
            device=self.device
        )

        # Step 1: Create 1D linspace for each dimension
        sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
        sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
        num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
        sample_event_type_min = prediction_config['probability_generation']['sample_event_type_min']
        sample_event_type_max = prediction_config['probability_generation']['sample_event_type_max']
        num_steps_event_type = prediction_config['probability_generation']['num_steps_event_type']
        time_since_last_event = torch.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime, device=self.device)
        event_types = torch.linspace(sample_event_type_min, sample_event_type_max, num_steps_event_type, device=self.device)

        # Step 2: Generate a 2D meshgrid
        time_grid, event_grid = torch.meshgrid(time_since_last_event, event_types, indexing="ij")

        # Step 3: Stack to get a grid of 2D samples with shape [num_samples, num_samples, 2]
        sample_grid = torch.stack((time_grid, event_grid), dim=-1)  # Shape: [num_samples, num_samples, 2]

        # Step 4: Reshape to get [num_samples * num_samples, 1, 1, 2]
        sample_grid = sample_grid.reshape(-1, 1, 1, 2)

        # Step 5: Expand to match [num_samples * num_samples, batch_size, seq_len, 2]
        sample_grid = sample_grid.expand(-1, batch_size, seq_len, -1)  # Shape: [num_samples * num_samples, batch_size, seq_len, 2]

        # Now sample_grid has the shape [num_samples * num_samples, batch_size, seq_len, 2]
        joint_logpdfs_pred = joint_dist.log_prob(sample_grid)

        # take the mean of the pred dist
        #joint_samples_pred = joint_dist.sample(( self.gen_config.num_sample_mean,))
        #joint_mean_pred = joint_samples_pred.mean(dim=0)
        #joint_mean_pred[:,:,0], joint_mean_pred[:,:,1],

        time_seq_label, time_delta_seq_label, event_seq_label, _, _ = batch
        return joint_logpdfs_pred, torch.empty(size=joint_logpdfs_pred.shape,device=self.device), time_delta_seq_label, event_seq_label
    

    def generate_samples_one_step_since_last_event(self, batch, prediction_config, forward=False):
        """One-step probabilities prediction for the last event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, _, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # time_delta_seq should start from 1, because the first one is zero
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        batch_size, seq_len = time_delta_seq[:, :-1].shape
        if not self.is_prior:
            # [batch_size, seq_len, hidden_size]
            context = self.forward(time_delta_seq[:, :-1], event_seq[:, :-1])

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = self.linear(context)
        else:
            # Unsqueeze to add batch and sequence dimensions
            # Shape: [1, 1, 6 * num_mix_components]
            expanded_linear = self.linear.unsqueeze(0).unsqueeze(0)  

            # Repeat the tensor across batch and sequence dimensions
            # Shape: [batch_size, seq_len, 6 * num_mix_components]
            expanded_linear = expanded_linear.repeat(batch_size, seq_len, 1)

            # [batch_size, seq_len, 6 * num_mix_components]
            raw_params = expanded_linear

        # Extract locs, log_scales, and log_weights from raw_params
        # locs: Means in 2D, so [batch_size, seq_len, num_components, 2]
        locs = raw_params[..., :self.num_mix_components * 2].reshape(batch_size, seq_len, self.num_mix_components, 2)

        # log_scales: Diagonal and off-diagonal elements of the covariance matrix
        # [batch_size, seq_len, num_components, 3]
        # 3 values per component in 2D: [variance_x, variance_y, covariance_xy]
        log_scales = raw_params[..., self.num_mix_components * 2: self.num_mix_components * 2 + self.num_mix_components * 3].reshape(batch_size, seq_len, self.num_mix_components, 3)

        # log_weights: Unchanged, normalized weights for each component
        log_weights = raw_params[..., (self.num_mix_components * 2 + self.num_mix_components * 3):]
        log_weights = torch.log_softmax(log_weights, dim=-1)

        log_scales = clamp_preserve_gradients(log_scales, -10.0, 3.0) # it was -5 to 3, but it was too small!
        joint_dist = MultivariateNormalMixtureDistribution(
            locs=locs, # [batch_size, seq_len, num_components, 3]
            log_scales=log_scales, # [batch_size, seq_len, num_components, 3]
            log_weights=log_weights, # [batch_size, seq_len, num_components]
            mean_inter_time=self.mean_inter_time,
            std_inter_time=self.std_inter_time,
            mean_event_type=self.mean_event_type,
            std_event_type=self.std_event_type,
            device=self.device
        )

        # Now sample_grid has the shape [num_samples * num_samples, batch_size, seq_len, 2]
        samples_pred = joint_dist.sample((prediction_config['num_samples_dtime']*prediction_config['num_samples_event_type'],))

        time_seq_label, time_delta_seq_label, event_seq_label, _, _ = batch
        return samples_pred, time_delta_seq_label, event_seq_label