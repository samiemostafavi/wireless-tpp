""" Base model with common functionality  """

from abc import abstractmethod
import torch
from torch import nn
from wireless_tpp.utils import set_device

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
        self.gen_config = model_config.thinning
        self.event_sampler = None
        self.device = set_device(model_config.gpu)
        self.use_mc_samples = model_config.use_mc_samples

        self.to(self.device)

    @abstractmethod
    def loglike_loss(self, batch):
        pass

    @abstractmethod
    def generate_samples_one_step_since_last_event(self, batch, prediction_config, forward=False):
        pass

    @abstractmethod
    def predict_probabilities_one_step_since_last_event(self, batch, prediction_config, forward=False):
        pass

    @abstractmethod
    def predict_multi_step_since_last_event(self, batch, prediction_config, forward=False):
        pass

    @abstractmethod
    def predict_one_step_at_every_event(self, batch, prediction_config, forward=False):
        pass


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

    