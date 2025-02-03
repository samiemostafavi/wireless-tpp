""" Initialize a Pytorch model wrapper that feed into Model Runner   """

import torch
from torch.utils.tensorboard import SummaryWriter

from wireless_tpp.utils import RunnerPhase, set_optimizer, set_device, logger

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

class TorchModelWrapper:
    def __init__(self, model, base_config, model_config, trainer_config, prediction_config):
        """A wrapper class for Torch backends.

        Args:
            model (BaseModel): a TPP model.
            base_config (EasyTPP.Config): basic configs.
            model_config (EasyTPP.ModelConfig): model spec configs.
            trainer_config (EasyTPP.TrainerConfig): trainer spec configs.
        """
        self.model = model
        self.base_config = base_config
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.prediction_config = prediction_config

        self.model_id = self.base_config.model_id
        self.device = set_device(self.trainer_config.gpu)

        self.model.to(self.device)

        if self.model_config.is_training:
            # set up optimizer
            optimizer = self.trainer_config.optimizer
            self.learning_rate = self.trainer_config.learning_rate
            self.opt = set_optimizer(optimizer, self.model.parameters(), self.learning_rate, self.trainer_config.weight_decay)
            logger.info(self.opt)

            # Initialize learning rate scheduler
            self.scheduler = ReduceLROnPlateau(
                self.opt,
                mode=self.trainer_config.lr_scheduler_mode, #min
                factor=self.trainer_config.lr_scheduler_factor, #0.6
                patience=self.trainer_config.lr_scheduler_patience #8
            )
            # Scaler for managing gradients
            self.scaler = GradScaler("cuda")

        # set up tensorboard
        self.use_tfb = self.trainer_config.use_tfb
        self.train_summary_writer, self.valid_summary_writer = None, None
        if self.use_tfb:
            self.train_summary_writer = SummaryWriter(log_dir=self.base_config.spec['tfb_train_dir'])
            self.valid_summary_writer = SummaryWriter(log_dir=self.base_config.spec['tfb_valid_dir'])

    def restore(self, ckpt_dir):
        """Load the checkpoint to restore the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """

        self.model.load_state_dict(torch.load(ckpt_dir), strict=False)

    def save(self, ckpt_dir):
        """Save the checkpoint for the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        torch.save(self.model.state_dict(), ckpt_dir)

    def write_summary(self, epoch, kv_pairs, phase):
        """Write the kv_paris into the tensorboard

        Args:
            epoch (int): epoch index in the training.
            kv_pairs (dict): metrics dict.
            phase (RunnerPhase): a const that defines the stage of model runner.
        """
        if self.use_tfb:
            summary_writer = None
            if phase == RunnerPhase.TRAIN:
                summary_writer = self.train_summary_writer
            elif phase == RunnerPhase.VALIDATE:
                summary_writer = self.valid_summary_writer
            elif phase == RunnerPhase.PREDICT:
                pass

            if summary_writer is not None:
                for k, v in kv_pairs.items():
                    if k != 'num_events':
                        summary_writer.add_scalar(k, v, epoch)

                summary_writer.flush()
        return

    def close_summary(self):
        """Close the tensorboard summary writer.
        """
        if self.train_summary_writer is not None:
            self.train_summary_writer.close()

        if self.valid_summary_writer is not None:
            self.valid_summary_writer.close()
        return

    def run_batch_mdn(self, batch, phase):
        """Run one batch.

        Args:
            batch (EasyTPP.BatchEncoding): preprocessed batch data that go into the model.
            phase (RunnerPhase): a const that defines the stage of model runner.

        Returns:
            tuple: for training and validation we return loss, prediction and labels;
            for prediction we return prediction.
        """

        batch = batch.to(self.device).values()

        # set mode to train
        is_training = (phase == RunnerPhase.TRAIN)
        self.model.train(is_training)

        # FullyRNN needs grad event in validation stage
        grad_flag = is_training
        # run model
        with torch.set_grad_enabled(grad_flag):
            with autocast(device_type="cuda"):
                loss, loss_mask = self.model.loglike_loss(batch, phase)

        # Assume we dont do prediction on train set
        pred_mean, pred_var, pred_q5a, pred_q5b, pred_q7a, pred_q7b, pred_q9a, pred_q9b, pred_q99a, pred_q99b, label, interarrival_time_seqs, len_seqs = None, None, None, None, None, None, None, None, None, None, None, None, None

        # update grad
        if is_training:
            self.opt.zero_grad()
            self.scaler.scale(loss.sum() / loss_mask.sum()).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        
            #self.opt.zero_grad()
            #(loss / num_event).backward()
            #self.opt.step()
        else:
            #self.model.eval()
            with torch.no_grad():
                (pred_mean, pred_var, pred_q5a, pred_q5b, pred_q7a, pred_q7b, pred_q9a, pred_q9b, pred_q99a, pred_q99b), label, \
                    (interarrival_time_seqs, len_seqs), pred_mask = self.model.predict(batch=batch)

                pred_mean = pred_mean.detach().cpu().numpy() if pred_mean is not None else None
                pred_var = pred_var.detach().cpu().numpy() if pred_var is not None else None
                pred_q5a = pred_q5a.detach().cpu().numpy() if pred_q5a is not None else None
                pred_q5b = pred_q5b.detach().cpu().numpy() if pred_q5b is not None else None
                pred_q7a = pred_q7a.detach().cpu().numpy() if pred_q7a is not None else None
                pred_q7b = pred_q7b.detach().cpu().numpy() if pred_q7b is not None else None
                pred_q9a = pred_q9a.detach().cpu().numpy() if pred_q9a is not None else None
                pred_q9b = pred_q9b.detach().cpu().numpy() if pred_q9b is not None else None
                pred_q99a = pred_q99a.detach().cpu().numpy() if pred_q99a is not None else None
                pred_q99b = pred_q99b.detach().cpu().numpy() if pred_q99b is not None else None
                label = label.detach().cpu().numpy() if label is not None else None

                interarrival_time_seqs = interarrival_time_seqs.detach().cpu().numpy() if interarrival_time_seqs is not None else None
                len_seqs = len_seqs.detach().cpu().numpy() if len_seqs is not None else None
                pred_mask = pred_mask.detach().cpu().numpy() if pred_mask is not None else None

            # check if pred_mask == loss_mask
            # TODO

        return loss.detach().cpu().numpy(), loss_mask.detach().cpu().numpy(), (pred_mean, pred_var, pred_q5a, pred_q5b, pred_q7a, pred_q7b, pred_q9a, pred_q9b, pred_q99a, pred_q99b), label, (interarrival_time_seqs, len_seqs)
