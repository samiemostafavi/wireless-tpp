from collections import OrderedDict
from pathlib import Path
import logging, json
import numpy as np
from wireless_tpp.utils import RunnerPhase, logger, MetricsHelper, MetricsTracker, concat_element, save_pickle, Timer, get_unique_id, LogConst, get_stage
from wireless_tpp.utils.const import Backend
from wireless_tpp.preprocess import TPPDataLoaderPacketArrival

# important: this line will register acc and rmse metrics
from wireless_tpp.default_registers.register_metrics import *


class TPPRunnerPacketArrival():
    """Standard TPP runner
    """

    def __init__(self, runner_config, unique_model_dir=False, **kwargs):
        """Initialize the runner.

        Args:
            runner_config (RunnerConfig): config for the runner.
            unique_model_dir (bool, optional): whether to give unique dir to save the model. Defaults to False.
        """

        self.eps = 1e-9
        
        self.runner_config = runner_config
        # re-assign the model_dir
        if unique_model_dir:
            runner_config.model_dir = runner_config.base_config.specs['saved_model_dir'] + '_' + get_unique_id()

        self.save_log()

        skip_data_loader = kwargs.get('skip_data_loader', False)
        if not skip_data_loader:
            # build data reader
            data_config = self.runner_config.data_config
            backend = self.runner_config.base_config.backend
            kwargs = self.runner_config.trainer_config.get_yaml_config()
            self._data_loader = TPPDataLoaderPacketArrival(
                data_config=data_config,
                backend=backend,
                **kwargs
            )

        # Needed for Intensity Free model
        mean_inter_time, std_inter_time, mean_event_type, std_event_type, min_dt, max_dt, min_eventtype, max_eventtype = (
            self._data_loader.train_loader().dataset.get_dt_stats()
        )
        runner_config.model_config.set("mean_inter_time", mean_inter_time)
        runner_config.model_config.set("std_inter_time", std_inter_time)
        runner_config.model_config.set("mean_log_inter_time", np.log(mean_inter_time+self.eps))
        runner_config.model_config.set("std_log_inter_time", np.log(std_inter_time+self.eps))
        runner_config.model_config.set("mean_event_type", mean_event_type)
        runner_config.model_config.set("std_event_type", std_event_type)
        runner_config.model_config.set("mean_log_event_type", np.log(mean_event_type+self.eps))
        runner_config.model_config.set("std_log_event_type", np.log(std_event_type+self.eps))
        self.timer = Timer()

        self.metrics_tracker = MetricsTracker()
        if self.runner_config.trainer_config.metrics is not None:
            self.metric_functions = self.runner_config.get_metric_functions()

        self._init_model()

        pretrain_dir = self.runner_config.model_config.pretrained_model_dir
        if pretrain_dir is not None:
            self._load_model(pretrain_dir)

    def _init_model(self):
        """Initialize the model.
        """
        self.use_torch = self.runner_config.base_config.backend == Backend.Torch

        if self.use_torch:
            from easy_tpp.utils import set_seed
            from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
            from easy_tpp.torch_wrapper import TorchModelWrapper
            from easy_tpp.utils import count_model_params
            set_seed(self.runner_config.trainer_config.seed)

            self.model = TorchBaseModel.generate_model_from_config(model_config=self.runner_config.model_config)
            self.model_wrapper = TorchModelWrapper(self.model,
                                                   self.runner_config.base_config,
                                                   self.runner_config.model_config,
                                                   self.runner_config.trainer_config,
                                                   self.runner_config.prediction_config
                                                   )
            num_params = count_model_params(self.model)

        else:
            from easy_tpp.utils.tf_utils import set_seed
            from easy_tpp.model.tf_model.tf_basemodel import TfBaseModel
            from easy_tpp.tf_wrapper import TfModelWrapper
            from easy_tpp.utils.tf_utils import count_model_params
            set_seed(self.runner_config.trainer_config.seed)

            self.model = TfBaseModel.generate_model_from_config(model_config=self.runner_config.model_config)
            self.model_wrapper = TfModelWrapper(self.model,
                                                self.runner_config.base_config,
                                                self.runner_config.model_config,
                                                self.runner_config.trainer_config)
            num_params = count_model_params()

        info_msg = f'Num of model parameters {num_params}'
        logger.info(info_msg)

    def _save_model(self, model_dir, **kwargs):
        """Save the model.

        Args:
            model_dir (str): the dir for model to save.
        """
        if model_dir is None:
            model_dir = self.runner_config.base_config.specs['saved_model_dir']
        self.model_wrapper.save(model_dir)
        logger.critical(f'Save model to {model_dir}')
        return

    def _load_model(self, model_dir, **kwargs):
        """Load the model from the dir.

        Args:
            model_dir (str): the dir for model to load.
        """
        self.model_wrapper.restore(model_dir)
        logger.critical(f'Load model from {model_dir}')
        return

    def get_config(self):
        return self.runner_config

    def set_model_dir(self, model_dir):
        self.runner_config.base_config.specs['saved_model_dir'] = model_dir

    def get_model_dir(self):
        return self.runner_config.base_config.specs['saved_model_dir']

    def train(
            self,
            train_loader=None,
            valid_loader=None,
            test_loader=None,
            **kwargs
    ):
        """Train the model.

        Args:
            train_loader (EasyTPP.DataLoader, optional): data loader for train set. Defaults to None.
            valid_loader (EasyTPP.DataLoader, optional): data loader for valid set. Defaults to None.
            test_loader (EasyTPP.DataLoader, optional): data loader for test set. Defaults to None.

        Returns:
            model: _description_
        """
        # no train and valid loader from outside
        if train_loader is None and valid_loader is None:
            train_loader = self._data_loader.train_loader()
            valid_loader = self._data_loader.valid_loader()

        # no test loader from outside and there indeed exits test data in config
        if test_loader is None and self.runner_config.data_config.test_dir is not None:
            test_loader = self._data_loader.test_loader()

        logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

        timer = self.timer
        timer.start()
        model_id = self.runner_config.base_config.model_id
        logger.info(f'Start {model_id} training...')
        model = self._train_model(
            train_loader,
            valid_loader,
            test_loader=test_loader,
            **kwargs
        )
        logger.info(f'End {model_id} train! Cost time: {timer.end()}')
        return model

    def evaluate(self, valid_loader=None, **kwargs):
        if valid_loader is None:
            valid_loader = self._data_loader.valid_loader()

        logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

        timer = self.timer
        timer.start()
        model_id = self.runner_config.base_config.model_id
        logger.info(f'Start {model_id} evaluation...')

        metric = self._evaluate_model(
            valid_loader,
            **kwargs
        )
        logger.info(f'End {model_id} evaluation! Cost time: {timer.end()}')
        return metric  # return a list of scalr for HPO to use

    def gen(self, gen_loader=None, **kwargs):
        if gen_loader is None:
            gen_loader = self._data_loader.test_loader()

        logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

        timer = self.timer
        timer.start()
        model_name = self.runner_config.base_config.model_id
        logger.info(f'Start {model_name} evaluation...')

        model = self._gen_model(
            gen_loader,
            **kwargs
        )
        logger.info(f'End {model_name} generation! Cost time: {timer.end()}')
        return model


    def save_log(self):
        """Save log to local files
        """
        log_dir = self.runner_config.base_config.specs['saved_log_dir']
        fh = logging.FileHandler(log_dir)
        fh.setFormatter(logging.Formatter(LogConst.DEFAULT_FORMAT_LONG))
        logger.addHandler(fh)
        logger.info(f'Save the log to {log_dir}')
        return

    def save(
            self,
            model_dir=None,
            **kwargs
    ):
        return self._save_model(model_dir, **kwargs)

    def run(self, **kwargs):
        """Start the runner.

        Args:
            **kwargs (dict): optional params.

        Returns:
            EasyTPP.BaseModel, dict: the results of the process.
        """
        current_stage = get_stage(self.runner_config.base_config.stage)
        if current_stage == RunnerPhase.TRAIN:
            return self.train(**kwargs)
        elif current_stage == RunnerPhase.VALIDATE:
            return self.evaluate(**kwargs)
        else:
            return self.gen(**kwargs)


    def _train_model(self, train_loader, valid_loader, **kwargs):
        """Train the model.

        Args:
            train_loader (EasyTPP.DataLoader): data loader for the train set.
            valid_loader (EasyTPP.DataLoader): data loader for the valid set.
        """
        test_loader = kwargs.get('test_loader')
        for i in range(self.runner_config.trainer_config.max_epoch):
            train_metrics = self.run_one_epoch(train_loader, RunnerPhase.TRAIN)

            message = f"[ Epoch {i} (train) ]: train " + MetricsHelper.metrics_dict_to_str(train_metrics)
            logger.info(message)

            self.model_wrapper.write_summary(i, train_metrics, RunnerPhase.TRAIN)

            # evaluate model
            if i % self.runner_config.trainer_config.valid_freq == 0:
                valid_metrics = self.run_one_epoch(valid_loader, RunnerPhase.VALIDATE)

                self.model_wrapper.write_summary(i, valid_metrics, RunnerPhase.VALIDATE)

                message = f"[ Epoch {i} (valid) ]:  valid " + MetricsHelper.metrics_dict_to_str(valid_metrics)
                logger.info(message)

                updated = self.metrics_tracker.update_best("loglike", valid_metrics['loglike'], i)

                message_valid = "current best loglike on valid set is {:.4f} (updated at epoch-{})".format(
                    self.metrics_tracker.current_best['loglike'], self.metrics_tracker.episode_best)

                if updated:
                    message_valid += f", best updated at this epoch"
                    self.model_wrapper.save(self.runner_config.base_config.specs['saved_model_dir'])

                if test_loader is not None:
                    test_metrics = self.run_one_epoch(test_loader, RunnerPhase.VALIDATE)

                    message = f"[ Epoch {i} (test) ]: test " + MetricsHelper.metrics_dict_to_str(test_metrics)
                    logger.info(message)

                logger.critical(message_valid)

        self.model_wrapper.close_summary()

        return

    def _evaluate_model(self, data_loader, **kwargs):
        """Evaluate the model on the valid dataset.

        Args:
            data_loader (EasyTPP.DataLoader): data loader for the valid set

        Returns:
            dict: metrics dict.
        """

        eval_metrics = self.run_one_epoch(data_loader, RunnerPhase.EVALUATE)

        self.model_wrapper.write_summary(0, eval_metrics, RunnerPhase.EVALUATE)

        self.model_wrapper.close_summary()

        message = f"Evaluation result: " + MetricsHelper.metrics_dict_to_str(eval_metrics)

        logger.critical(message)

        # save it to a json file
        model_dir = self.runner_config.base_config.specs['log_folder']
        logger.critical(f'Save evaluation results to {Path(model_dir) / "eval.json"}')

        # Convert numpy types to Python native types
        eval_metrics = {key: float(value) if isinstance(value, np.floating) else value for key, value in eval_metrics.items()}

        # save json file
        with open(Path(model_dir) / 'eval.json', 'w') as file:
            json.dump(eval_metrics, file, indent=4)

        return eval_metrics

    def _gen_model(self, data_loader, **kwargs):
        """Generation of the TPP, one-step and multi-step are both supported.
        """
        
        if kwargs.get('probability_generation', False):
            test_result = self.run_one_epoch_probability_generation(data_loader, RunnerPhase.PREDICT)
        else:
            test_result = self.run_one_epoch_sample_generation(data_loader, RunnerPhase.PREDICT)
            
        # save it to a pkl file
        model_dir = self.runner_config.base_config.specs['log_folder']
        logger.critical(f'Save prediction results to {Path(model_dir) / "pred.pkl"}')
        save_pickle(Path(model_dir) / 'pred.pkl', test_result)
        return

    def run_one_epoch(self, data_loader, phase):
        """Run one complete epoch.

        Args:
            data_loader: data loader object defined in model runner
            phase: enum, [train, dev, test]

        Returns:
            a dict of metrics
        """
        total_loss = 0
        dtime_loss = 0
        event_loss = 0
        total_num_event = 0
        epoch_label = []
        epoch_pred = []
        epoch_mask = []
        pad_index = self.runner_config.data_config.data_specs.pad_token_id
        metrics_dict = OrderedDict()
        if phase in [RunnerPhase.TRAIN, RunnerPhase.VALIDATE]:
            for batch in data_loader:
                batch_loss, batch_num_event, batch_pred, batch_label, batch_mask = \
                    self.model_wrapper.run_batch(batch, phase=phase)

                total_loss += batch_loss
                total_num_event += batch_num_event
                epoch_pred.append(batch_pred)
                epoch_label.append(batch_label)
                epoch_mask.append(batch_mask)

            avg_loss = total_loss / total_num_event

            metrics_dict.update({'loglike': -avg_loss, 'num_events': total_num_event})

        else:
            for batch in data_loader:
                batch_pred, ll_dtime, ll_type, batch_num_event, batch_label = self.model_wrapper.run_batch(batch, phase=phase)
                total_loss += (ll_dtime+ll_type)
                dtime_loss += ll_dtime
                event_loss += ll_type
                total_num_event += batch_num_event
                epoch_pred.append(batch_pred)
                epoch_label.append(batch_label)

            avg_total_ll = total_loss / total_num_event
            avg_dtime_ll = dtime_loss / total_num_event
            avg_event_ll = event_loss / total_num_event

            metrics_dict.update({'total_ll': avg_total_ll, 'dtime_ll':avg_dtime_ll, 'event_type_ll':avg_event_ll, 'num_events': total_num_event})

        # we need to improve the code here
        # classify batch_output to list
        #pred_exists, label_exists = False, False
        #if epoch_pred[0][0] is not None:
        #    epoch_pred = concat_element(epoch_pred, pad_index)
        #    pred_exists = True
        #if len(epoch_label) > 0 and epoch_label[0][0] is not None:
        #    epoch_label = concat_element(epoch_label, pad_index)
        #    label_exists = True
        #    if len(epoch_mask):
        #        epoch_mask = concat_element(epoch_mask, False)[0]  # retrieve the first element of concat array
        #        epoch_mask = epoch_mask.astype(bool)
        #if pred_exists and label_exists:
        #    metrics_dict.update(self.metric_functions(epoch_pred, epoch_label, seq_mask=epoch_mask))

        if self.runner_config.base_config.model_id == 'IntensityFree':
            metrics_dict.update(
                {
                    'rmse_dtime' : rmse_dtime_metric_function(epoch_pred, epoch_label),
                    'rmse_event' : rmse_event_metric_function(epoch_pred, epoch_label),
                }
            )
        elif self.runner_config.base_config.model_id == 'IntensityFree2D':
            metrics_dict.update(
                {
                    'rmse_2d' : rmse_2d_metric_function(epoch_pred, epoch_label),
                    'rmse_2d_dtime' : rmse_2d_dtime_metric_function(epoch_pred, epoch_label),
                    'rmse_2d_event' : rmse_2d_event_metric_function(epoch_pred, epoch_label),
                }
            )
        else:
            metrics_dict.update(self.metric_functions(epoch_pred, epoch_label, seq_mask=epoch_mask))

        if phase == RunnerPhase.PREDICT:
            metrics_dict.update({'pred': epoch_pred, 'label': epoch_label})

        return metrics_dict
    
    def run_one_epoch_probability_generation(self, data_loader, phase):
        """Run one complete epoch and store the intensity values.

        Args:
            data_loader: data loader object defined in model runner
            phase: enum, [train, dev, test]

        Returns:
            a dict of results
        """

        probs_pred = []
        epoch_label = []
        metrics_dict = OrderedDict()
        if phase is not RunnerPhase.PREDICT:
            return
        
        for batch in data_loader:
            batch_probs, batch_label = self.model_wrapper.run_batch_probability_generation(batch, phase=phase)
            probs_pred.append(batch_probs)
            epoch_label.append(batch_label)

        if phase == RunnerPhase.PREDICT:
            metrics_dict.update({'pred': probs_pred, 'label': epoch_label})

        return metrics_dict
    
    def run_one_epoch_sample_generation(self, data_loader, phase):
        """Run one complete epoch and store the intensity values.

        Args:
            data_loader: data loader object defined in model runner
            phase: enum, [train, dev, test]

        Returns:
            a dict of results
        """

        samples_pred = []
        epoch_label = []
        metrics_dict = OrderedDict()
        if phase is not RunnerPhase.PREDICT:
            return
        
        for batch in data_loader:
            batch_samples, batch_label = self.model_wrapper.run_batch_sample_generation(batch, phase=phase)
            samples_pred.append(batch_samples)
            epoch_label.append(batch_label)

        if phase == RunnerPhase.PREDICT:
            metrics_dict.update({'pred': samples_pred, 'label': epoch_label})

        return metrics_dict

