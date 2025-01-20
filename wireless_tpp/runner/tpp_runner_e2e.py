from collections import OrderedDict
from pathlib import Path
import logging, json
import numpy as np
from wireless_tpp.utils import RunnerPhase, logger, MetricsHelper, MetricsTracker, concat_element, save_pickle, Timer, get_unique_id, LogConst, get_stage
from wireless_tpp.utils.const import Backend
from wireless_tpp.preprocess.scheduling import TPPDataLoaderScheduling

# important: this line will register acc and rmse metrics
from wireless_tpp.default_registers.register_metrics import *


class TPPRunnerE2E():
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

        disable_logging = kwargs.get('disable_logging', False)
        if not disable_logging:
            self.save_log()

        # build data reader
        data_config = self.runner_config.data_config
        backend = self.runner_config.base_config.backend
        kwargs = self.runner_config.trainer_config.get_yaml_config()
        self._data_loader = TPPDataLoaderScheduling(
            data_config=data_config,
            backend=backend,
            source_data=None,
            **kwargs
        )

        # needed for transformation of the data 
        current_stage = get_stage(self.runner_config.base_config.stage)
        if data_config is not None and current_stage == RunnerPhase.TRAIN:
            mean_dtime, std_dtime, mean_event_type, std_event_type, min_dt, max_dt, min_eventtype, max_eventtype = (
                self._data_loader.train_loader().dataset.get_stats(inp_type='time_delta_seqs', packet_or_segment=True, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            )
            runner_config.model_config.model_specs["mean_dtime"] = float(mean_dtime)
            runner_config.model_config.model_specs["std_dtime"] = float(std_dtime)
            
            mean_len, std_len, mean_event_type, std_event_type, min_len, max_len, min_eventtype, max_eventtype = (
                self._data_loader.train_loader().dataset.get_stats(inp_type='len_seqs', packet_or_segment=False, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            )
            runner_config.model_config.model_specs["mean_len"] = float(mean_len)
            runner_config.model_config.model_specs["std_len"] = float(std_len)

            mean_interarrival_time, std_interarrival_time, mean_event_type, std_event_type, min_interarrival_time, max_interarrival_time, min_eventtype, max_eventtype = (
                self._data_loader.train_loader().dataset.get_stats(inp_type='interarrival_time_seqs', packet_or_segment=True, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            )
            runner_config.model_config.model_specs["mean_interarrival_time"] = float(mean_interarrival_time)
            runner_config.model_config.model_specs["std_interarrival_time"] = float(std_interarrival_time)

            # just for the sake of reporting in logs
            self._data_loader.train_loader().dataset.get_stats(inp_type='label_mask_seqs', packet_or_segment=True, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='len_seqs', packet_or_segment=False, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='mcs_seqs', packet_or_segment=False, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='mretx_seqs', packet_or_segment=False, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='rfailed_seqs', packet_or_segment=False, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='len_seqs', packet_or_segment=True, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='mcs_seqs', packet_or_segment=True, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='mretx_seqs', packet_or_segment=True, num_event_types=self.runner_config.data_config.data_specs.num_event_types)
            self._data_loader.train_loader().dataset.get_stats(inp_type='rfailed_seqs', packet_or_segment=True, num_event_types=self.runner_config.data_config.data_specs.num_event_types)

            # save again to save the updated config
            self.runner_config.save_config()

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

        from wireless_tpp.utils import set_seed
        from wireless_tpp.model.basemodel import TorchBaseModel
        from wireless_tpp.torch_wrapper import TorchModelWrapper
        from wireless_tpp.utils import count_model_params
        set_seed(self.runner_config.trainer_config.seed)

        self.model = TorchBaseModel.generate_model_from_config(model_config=self.runner_config.model_config)
        self.model_wrapper = TorchModelWrapper(self.model,
                                                self.runner_config.base_config,
                                                self.runner_config.model_config,
                                                self.runner_config.trainer_config,
                                                self.runner_config.prediction_config
                                                )
        num_params = count_model_params(self.model)

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

    def evaluate(self, test_loader=None, **kwargs):
        if test_loader is None:
            test_loader = self._data_loader.test_loader()

        logger.info(f'Data \'{self.runner_config.base_config.dataset_id}\' loaded...')

        timer = self.timer
        timer.start()
        model_id = self.runner_config.base_config.model_id
        logger.info(f'Start {model_id} evaluation...')

        metric = self._evaluate_model(
            test_loader,
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
        source_data = kwargs.get('source_data', None)
        if source_data is not None:
            source_data_specs = kwargs.get('data_specs')
            data_config = self.runner_config.data_config
            backend = self.runner_config.base_config.backend
            # {'seed': 2019, 'gpu': -1, 'batch_size': 1, 'max_epoch': 800, 'shuffle': False, 'optimizer': 'adam', 'learning_rate': 0.0001, 'valid_freq': 10, 'use_tfb': False, 'metrics': ['acc', 'rmse']}
            kwargs_train = self.runner_config.trainer_config.get_yaml_config()
            batch_size = kwargs.get('batch_size', None)
            if batch_size is not None:
                kwargs_train['batch_size'] = batch_size
            self._data_loader = TPPDataLoaderScheduling(
                data_config=data_config,
                backend=backend,
                source_data=source_data,
                source_data_specs=source_data_specs,
                **kwargs_train
            )

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
                #valid_metrics = self.run_one_epoch(valid_loader, RunnerPhase.VALIDATE)
                valid_metrics = self.run_one_epoch(test_loader, RunnerPhase.VALIDATE)

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
        
        if kwargs.get('return_predictions', False):
            return test_result
        else:
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
        total_dtime_error = 0
        total_num_event = 0
        total_dtime_var = 0
        sum_70 = 0.0
        sum_90 = 0.0
        sum_99 = 0.0
        sum_999 = 0.0
        epoch_label = []
        epoch_pred = []
        epoch_pred_var = []
        metrics_dict = OrderedDict()

        for batch in data_loader:
            batch_loss, batch_num_event, batch_pred, batch_label, batch_mask, _, _, batch_pred_var, batch_pred_quantile = \
                self.model_wrapper.run_batch_mdn(batch, phase=phase)
            total_loss += batch_loss
            total_num_event += batch_num_event
            if phase == RunnerPhase.VALIDATE or phase == RunnerPhase.EVALUATE:
                # assert the shape
                # are only one dimensional like: (batch_size, )
                assert batch_label[0].shape == batch_pred[0].shape == batch_pred_var[0].shape, \
                    "Shapes of batch_label, batch_pred, batch_pred_var, must be the same {} {} {}".format(
                        batch_label[0].shape, batch_pred[0].shape, batch_pred_var[0].shape
                    )
                assert batch_mask.shape == batch_label[0].shape
                tmp = np.array(abs(batch_label[0] - batch_pred[0]))*batch_mask
                total_dtime_error += sum(sum(tmp))
                total_dtime_var += sum(sum(np.array(batch_pred_var[0])))
                batch_pred_q7 = batch_pred_quantile[0]
                batch_pred_q9 = batch_pred_quantile[1]
                batch_pred_q99 = batch_pred_quantile[2]
                batch_pred_q999 = batch_pred_quantile[3]
                sum_70 += (np.array(batch_label[0]) <= np.array(batch_pred_q7)).sum()
                sum_90 += (np.array(batch_label[0]) <= np.array(batch_pred_q9)).sum()
                sum_99 += (np.array(batch_label[0]) <= np.array(batch_pred_q99)).sum()
                sum_999 += (np.array(batch_label[0]) <= np.array(batch_pred_q999)).sum()
                epoch_pred.append(batch_pred)
                epoch_pred_var.append(batch_pred_var)
                epoch_label.append(batch_label)

        # calc loss
        avg_loss = total_loss / total_num_event
        metrics_dict.update({'loglike': -avg_loss, 'num_events': total_num_event})

        if phase == RunnerPhase.VALIDATE:
            if hasattr(self.model_wrapper, "scheduler"):
                # Use validation metric to step the scheduler
                self.model_wrapper.scheduler.step(-avg_loss)

                current_lr = self.model_wrapper.opt.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")

        # calc errors
        if phase == RunnerPhase.VALIDATE or phase == RunnerPhase.EVALUATE:
            coverage_70 = sum_70 / total_num_event
            coverage_90 = sum_90 / total_num_event
            coverage_99 = sum_99 / total_num_event
            coverage_999 = sum_999 / total_num_event
            avg_dtime_error = total_dtime_error / total_num_event
            avg_dtime_var = total_dtime_var / total_num_event
            metrics_dict.update({'dtime_mae': avg_dtime_error, 'dtime_var': avg_dtime_var, 'coverage_70': coverage_70, 'coverage_90': coverage_90, 'coverage_99': coverage_99, 'coverage_999': coverage_999})

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
        masks = []
        metrics_dict = OrderedDict()
        if phase is not RunnerPhase.PREDICT:
            return
        
        for batch in data_loader:
            batch_probs, batch_label, batch_masks = self.model_wrapper.run_batch_probability_generation_scheduling(batch, phase=phase)
            probs_pred.append(batch_probs)
            epoch_label.append(batch_label)
            masks.append(batch_masks)

        metrics_dict.update({'pred': probs_pred, 'label': epoch_label, 'mask': masks})

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
        masks = []
        metrics_dict = OrderedDict()
        if phase is not RunnerPhase.PREDICT:
            return
        
        for batch in data_loader:
            batch_samples, batch_label, batch_mask = self.model_wrapper.run_batch_sample_generation_scheduling(batch, phase=phase)
            samples_pred.append(batch_samples)
            epoch_label.append(batch_label)
            masks.append(batch_mask)

        if phase == RunnerPhase.PREDICT:
            metrics_dict.update({'pred': samples_pred, 'label': epoch_label, 'mask': masks})

        return metrics_dict