import torch
import torch.distributions as D
from torch import nn
from torch.distributions import Categorical, TransformedDistribution
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions import Normal as TorchNormal
from torch.distributions import MultivariateNormal as MultivariateTorchNormal



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
        mean_log_val (float): Average log-inter-event-time.
        std_log_val (float): Std of log-inter-event-times.
    """

    def __init__(self, locs, log_scales, log_weights, mean_val, std_val, validate_args=None):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=torch.exp(log_scales))
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_val == 0.0 and std_val == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_val, scale=std_val)]

        self.mean_val = mean_val
        self.std_val = std_val

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
        
    @property
    def locs(self):
        return self.base_dist.component_distribution.loc
    @property
    def scales(self):
        return self.base_dist.component_distribution.scale
    @property
    def weights(self):
        return torch.softmax(self.base_dist.mixture_distribution.logits, dim=-1)
    
    @property
    def mean(self):
        """
        Returns the mean of the distribution in the *transformed* (observed) space.
        Shape: [batch_size, seq_len].
        """
        # 1) Extract mixture components in base space
        mixture_dist = self.base_dist.mixture_distribution  # Categorical
        component_dist = self.base_dist.component_distribution  # Normal

        # => shape [batch_size, seq_len, num_components]
        weights = torch.softmax(mixture_dist.logits, dim=-1)          # w_k
        locs = component_dist.loc                                     # μ_k
        scales = component_dist.scale                                 # σ_k

        # 2) Mean in base space
        # [batch_size, seq_len] after summation
        base_mean = (weights * locs).sum(dim=-1)

        # 3) If there's an affine transform, apply it
        #    We only expect 0 or 1 transforms from your code, but let's be general
        y_mean = base_mean
        for transform in self.transforms:
            # transform is D.AffineTransform(loc, scale)
            # y = scale * x + loc
            # So E[Y] = scale * E[X] + loc
            y_mean = transform.scale * y_mean + transform.loc

        return y_mean

    @property
    def variance(self):
        """
        Returns the variance of the distribution in the *transformed* (observed) space.
        Shape: [batch_size, seq_len].
        """
        # 1) Extract mixture components in base space
        mixture_dist = self.base_dist.mixture_distribution
        component_dist = self.base_dist.component_distribution

        weights = torch.softmax(mixture_dist.logits, dim=-1)  # [B, S, K]
        locs = component_dist.loc                             # [B, S, K]
        scales = component_dist.scale                         # [B, S, K]

        # 2) Compute mean and second moment in base space
        base_mean = (weights * locs).sum(dim=-1)  # [B, S]

        # E[X^2] = sum_k w_k (σ_k^2 + μ_k^2)
        # So second moment = E[X^2].
        second_moment = (weights * (scales.pow(2) + locs.pow(2))).sum(dim=-1)  # [B, S]

        # Var[X] = E[X^2] - (E[X])^2
        base_var = second_moment - base_mean.pow(2)  # [B, S]

        # 3) Apply transforms
        # For each AffineTransform y = scale * x + loc:
        # Var[Y] = scale^2 * Var[X]
        # If multiple transforms, multiply all scales^2
        y_var = base_var
        total_scale = 1.0
        for transform in self.transforms:
            total_scale = total_scale * transform.scale
        # Multiply once
        y_var = y_var * (total_scale ** 2)

        return y_var


class NormalMixtureDistribution2D(TransformedDistribution):
    """
    Mixture of multivariate normal distributions.

    Args:
        locs (tensor): [batch_size, seq_len, num_mix_components].
        log_scales (tensor): [batch_size, seq_len, num_mix_components].
        log_weights (tensor): [batch_size, seq_len, num_mix_components].
        mean_log_inter_time (float): Average log-inter-event-time.
        std_log_inter_time (float): Std of log-inter-event-times.
    """

    def __init__(self, locs, log_scales, log_weights, mean_dtime, std_dtime, mean_len, std_len, device, validate_args=None):
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

        if mean_dtime == 0.0 and std_dtime == 1.0 and mean_len == 0.0 and std_len == 1.0:
            transforms = []
        else:
            mean_2d = torch.tensor([mean_dtime, mean_len], device=self.device)
            std_2d = torch.tensor([std_dtime, std_len], device=self.device)
            transforms = [D.AffineTransform(loc=mean_2d, scale=std_2d)]

        self.mean_2d = [mean_dtime, mean_len]
        self.std_2d = [std_dtime, std_len]
        self.mean_dtime = mean_dtime
        self.std_dtime = std_dtime
        self.mean_len = mean_len
        self.std_len = std_len

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

    def marginalize(self, dim):
        """
        Marginalize over one dimension of the 2D distribution.

        Args:
            dim (int): Dimension to marginalize out (integrate over).
                       0 for 'dtime' (first dimension), 
                       1 for 'len' (second dimension).

        Returns:
            MixtureSameFamily: A new mixture distribution over the remaining dimension
                               **in the same (transformed) space** as the original.
        """
        # -----------------------------
        # 1) Get the base 2D mixture
        # -----------------------------
        # base_dist is MixtureSameFamily(MultivariateNormal(...))
        mixture_dist = self.base_dist.mixture_distribution
        mvn_components = self.base_dist.component_distribution  # MultivariateNormal

        # -----------------------------
        # 2) Extract the marginal in *base* space (no transforms yet)
        # -----------------------------
        # For a 2D MVN with mean=[mu_x, mu_y], scale_tril=
        #  [[sigma_x,        0],
        #   [rho*sigma_x,  sigma_y]],
        # the marginal for dimension 'out_dim' is simply Normal(mu_out_dim, sigma_out_dim).
        loc_base = mvn_components.loc[..., dim]  # shape: [..., num_components]
        scale_base = mvn_components.scale_tril[..., dim, dim]  # [..., num_components]

        # -----------------------------
        # 3) Build the new 1D mixture distribution
        # -----------------------------
        # Same mixture weights, but components are now univariate Normal
        new_mixture = NormalMixtureDistribution(
            locs=loc_base, 
            log_scales=torch.log(scale_base), 
            log_weights=mixture_dist.logits,
            mean_val=self.mean_2d[dim],
            std_val=self.std_2d[dim],
        )
        return new_mixture


    @property
    def mean(self):
        """
        Returns the [batch_size, seq_len, 2] mean of the distribution 
        in the *transformed* (observed) space.
        """
        # 1) Access base mixture
        mixture_dist = self.base_dist.mixture_distribution   # Categorical
        mvn_components = self.base_dist.component_distribution  # MultivariateNormal

        weights = torch.softmax(mixture_dist.logits, dim=-1)     # [B, S, K]
        locs = mvn_components.loc                                # [B, S, K, 2]
        # scale_tril = mvn_components.scale_tril                 # [B, S, K, 2, 2] if needed

        # 2) Mean in base space
        base_mean = (weights.unsqueeze(-1) * locs).sum(dim=-2)    # [B, S, 2]

        # 3) Apply each AffineTransform
        y_mean = base_mean
        for transform in self.transforms:
            # transform.loc, transform.scale are shape [2]
            # Y = loc + scale * X (element-wise)
            y_mean = transform.loc + transform.scale * y_mean

        return y_mean

    @property
    def covariance(self):
        """
        Returns the [batch_size, seq_len, 2, 2] covariance in the 
        *transformed* (observed) space.
        """
        mixture_dist = self.base_dist.mixture_distribution
        mvn_components = self.base_dist.component_distribution
        
        weights = torch.softmax(mixture_dist.logits, dim=-1)        # [B, S, K]
        locs = mvn_components.loc                                   # [B, S, K, 2]
        scale_tril = mvn_components.scale_tril                      # [B, S, K, 2, 2]

        # 1) Compute base-space mean: [B, S, 2]
        base_mean = (weights.unsqueeze(-1) * locs).sum(dim=-2)

        # 2) Sum up: Cov(X) = \sum_k w_k [Sigma_k + mu_k mu_k^T] - mean(X) mean(X)^T
        #    Where Sigma_k = scale_tril_k @ scale_tril_k^T
        # We'll do it in a loop or carefully with broadcasting.
        # shape details:
        #   scale_tril[..., k, 2, 2] => Sigma_k = scale_tril[..., k] @ scale_tril[..., k].T

        # Let's compute each Sigma_k:
        # We do a batched matmul to get Sigma_k = scale_tril_k * scale_tril_k^T
        # shape: [B, S, K, 2, 2]
        Sigma_k = scale_tril @ scale_tril.transpose(-1, -2)  

        # mu_k outer product mu_k => [B, S, K, 2, 2]
        mu_k_outer = locs.unsqueeze(-1) * locs.unsqueeze(-2)  # (x,y) outer => 2x2

        # Sigma_k + mu_k outer
        comp_contrib = Sigma_k + mu_k_outer

        # Weighted sum over K
        # weights shape => [B, S, K], so we can expand to [B, S, K, 1, 1]
        w_expand = weights.unsqueeze(-1).unsqueeze(-1)
        sum_comp_contrib = (w_expand * comp_contrib).sum(dim=-3)  # [B, S, 2, 2]

        # base_mean outer => [B, S, 2, 2]
        base_mean_outer = base_mean.unsqueeze(-1) * base_mean.unsqueeze(-2)

        base_cov = sum_comp_contrib - base_mean_outer  # [B, S, 2, 2]

        # 3) Apply the AffineTransform => Cov(Y) = A Cov(X) A^T
        # For transform.scale shape [2], A = diag(scale_x, scale_y).
        # We'll accumulate the product of scale factors in a diagonal matrix.

        # We'll combine all transforms into one effective diagonal scale, e.g. scale = scale_1 * scale_2 ...
        scale_x, scale_y = 1.0, 1.0
        for transform in self.transforms:
            scale_x *= transform.scale[0]
            scale_y *= transform.scale[1]

        # Construct A = [[scale_x,      0],
        #                [0,      scale_y]]
        # Cov(Y) = A Cov(X) A^T => just elementwise multiply for diagonal
        # (A Cov(X)) => scale_x^2 for [0,0] and scale_y^2 for [1,1], no cross terms if it's diagonal.

        # shape [B, S, 2, 2]
        # base_cov[..., 0, 0] *= scale_x^2
        # base_cov[..., 0, 1] *= scale_x*scale_y
        # base_cov[..., 1, 0] *= scale_x*scale_y
        # base_cov[..., 1, 1] *= scale_y^2

        # A convenient approach:
        diag_scales = torch.tensor([scale_x, scale_y], device=base_cov.device)
        # We'll do: Cov(Y) = diag_scales * Cov(X) * diag_scales, with broadcasting
        # but we have to do it carefully for the 2D matrix shape.

        # One-liner: base_cov = diag_scales.unsqueeze(-1)*base_cov*diag_scales
        # shape => [2], so expand to [1,1,2], etc. might need to do some shape manipulation:
        diag_scales_ = diag_scales.view(1, 1, 2)  # shape [1,1,2]
        # Now multiply:
        # multiply rows
        scaled_cov = base_cov * diag_scales_.unsqueeze(-2)  # [B,S,2,2] * [1,1,1,2]
        # multiply columns
        scaled_cov = scaled_cov * diag_scales_.unsqueeze(-1) # [1,1,2,1]

        return scaled_cov


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cpu'):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
