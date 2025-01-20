import torch
import torch.distributions as D
from torch import nn
import math

from wireless_tpp.utils import logger

from wireless_tpp.model.mdn import clamp_preserve_gradients, NormalMixtureDistribution2D, NormalMixtureDistribution, AddGaussianNoise


class MixtureDistribution(nn.Module):
    """Mixture distribution for the delay prediction.
    Applies noise regularization to the delay values.

    Args:

        model_config: dict, model configuration
    """

    def __init__(self, model_config, device):
        super(MixtureDistribution, self).__init__()
        self.num_mix_components = model_config.model_specs['mdn']['num_mix_components']
        self.mean_dtime = model_config.model_specs.get("mean_dtime", 0.0)
        self.std_dtime = model_config.model_specs.get("std_dtime", 1.0)

        self.device = device

        # Noise regularization, only Gaussian noise is supported
        if model_config.noise_regularization.dtime['noise_type'] == 'gaussian':
            logger.info(f"Add Gaussian noise to dtime with std dev: {model_config.noise_regularization.dtime['std_dev']}")
            self.nr_dtime = AddGaussianNoise(mean=0, std=model_config.noise_regularization.dtime['std_dev'], device=self.device)
        else:
            self.nr_dtime = AddGaussianNoise(mean=0, std=0, device=self.device)

    @property
    def num_params(self):
        # GMM only
        return 3 * self.num_mix_components

    def forward(self, raw_params) -> NormalMixtureDistribution:
        """Compute the distribution of delta time.

        Args:

        raw_params (tensor): [batch_size, seq_len, 3 * self.num_mix_components]

        Returns:
            NormalMixtureDistribution: delay distribution with dim: [batch_size, seq_len]
        """
        assert raw_params.size(2) == self.num_params

        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]
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
    
    def negative_loglikelihood(self, raw_params, labels, pad_mask):
        """Compute the negative log-likelihood of the target labels.

        Args:
            raw_params (tensor): [batch_size, seq_len, 3 * self.num_mix_components]
            labels (tensor): [batch_size, seq_len]
            pad_mask (tensor): [batch_size, seq_len] indicating the valid positions (1) and padding positions (0)
            forward (bool): whether to compute the forward pass

        Returns:
            tensor: negative log-likelihood [batch_size]
            tensor: number of predictions [batch_size]
        """
        pad_mask = pad_mask.float()
        
        assert raw_params.size(2) == self.num_params
        assert raw_params.size(0) == labels.size(0)
        assert raw_params.size(1) == labels.size(1)

        labels = self.nr_dtime(labels)

        pred_dist = self.forward(raw_params)

        # Apply prediction mask to filter out invalid positions
        assert labels.shape == pred_dist.mean.shape # [batch_size, seq_len]
        dtime_ll = pred_dist.log_prob(labels) * pad_mask
        nll = -dtime_ll.sum()

        num_predictions = pad_mask.sum()
        return nll, num_predictions
    

    def mean_variance(self, raw_params):
        """Compute the mean and variance of the target labels.

        Args:
            raw_params (tensor): [batch_size, seq_len, 3 * self.num_mix_components]

        Returns:
            tensor: mean predictions [batch_size]
            tensor: variance of the predictions [batch_size]
        """
        assert raw_params.size(2) == self.num_params
        pred_dist = self.forward(raw_params)

        return pred_dist.mean, pred_dist.variance
    

    def quantile(self, raw_params, q):
        """Compute the quantile of the target labels.

        Args:
            raw_params (tensor): [batch_size, seq_len, 3 * self.num_mix_components]
            q (float): quantile value

        Returns:
            tensor: quantile predictions [batch_size]
        """
        assert raw_params.size(2) == self.num_params
        pred_dist = self.forward(raw_params)

        # 1) Extract mixture parameters
        locs = pred_dist.locs       # [B, T, K]
        scales = pred_dist.scales   # [B, T, K]
        weights = pred_dist.weights # [B, T, K]

        # 2) Call the bisection or any numeric root finder
        x_q = mixture_quantile(locs, scales, weights, q)
        x_q_transformed = pred_dist.transforms[0](x_q)

        # 3) Return the qth-quantile
        return x_q_transformed
    

def normal_cdf(z):
    """Compute standard normal CDF for tensor z using erf."""
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

def mixture_cdf(x, locs, scales, weights):
    """
    x:      [batch_size, seq_len]   (the point at which we evaluate the CDF)
    locs:   [batch_size, seq_len, K]
    scales: [batch_size, seq_len, K]
    weights:[batch_size, seq_len, K]
    Returns:
      cdf_val: [batch_size, seq_len], the mixture's CDF evaluated at x
    """
    # Expand x to shape [batch_size, seq_len, 1] so it can broadcast over K
    x_expanded = x.unsqueeze(-1)  # [B, T, 1]
    z = (x_expanded - locs) / scales  # [B, T, K]
    cdf_vals = normal_cdf(z)         # elementwise standard normal CDF
    mix_cdf = (cdf_vals * weights).sum(dim=-1)  # sum over mixture components
    return mix_cdf

def mixture_quantile(locs, scales, weights, q, max_iter=30):
    """
    Numerically find the quantile x_q s.t. mixture_cdf(x_q) = q.

    locs, scales, weights each shape: [B, T, K]
    q in (0,1): the quantile
    Returns:
      x_q: [B, T], the q-th quantile for each mixture distribution
    """
    # We’ll create a bracket [left, right] that definitely contains the quantile.
    # A simple approach is to take min(locs) - 5 * max(scales) as left,
    # and max(locs) + 5 * max(scales) as right, per (B,T) distribution.
    left = locs.amin(dim=-1) - 5.0 * scales.amax(dim=-1)  # [B, T]
    right = locs.amax(dim=-1) + 5.0 * scales.amax(dim=-1) # [B, T]

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        cdf_val = mixture_cdf(mid, locs, scales, weights)  # [B, T]
        # Where cdf_val < q, we move left up, otherwise move right down
        mask = (cdf_val < q)
        left = torch.where(mask, mid, left)
        right = torch.where(mask, right, mid)

    # After ~30 iterations, left ~ right => the quantile
    x_q = 0.5 * (left + right)
    return x_q