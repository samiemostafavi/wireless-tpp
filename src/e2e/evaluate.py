import plotly.graph_objects as go
from pathlib import Path
import yaml, pickle, json
import numpy as np

from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerE2E
from wireless_tpp.utils import logger


def evaluate_model(args):
    
    # read configuration from args.config
    prediction_config_path = Path(args.config)
    with open(prediction_config_path, 'r') as f:
        prediction_config = json.load(f)
    prediction_config = prediction_config[args.configname]
    batch_size = prediction_config['batch_size']
    gpu = prediction_config['gpu']

    if args.id:
        model_path = Path(args.source) / "e2e" / "trained_models" / args.name / args.id
    else:
        # if no id is passed, take the first folder in the trained_models directory
        tranied_models_path = Path(args.source) / "e2e" / "trained_models" / args.name
        model_path = next(tranied_models_path.iterdir())

    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    logger.info(f"Loaded model {model_path}")

    if prediction_config['dataset']:
        dataset_path = Path(prediction_config['dataset']) / "e2e" / "datasets" / prediction_config['dataset_name']
    else:
        if prediction_config['dataset_name']:
            dataset_path = Path(args.source) / "e2e" / "datasets" / prediction_config['dataset_name']
        else:
            # take the default test dataset of the training dataset
            dataset_path = Path(training_output_config['data_config']['train_dir']).parent

    dataset_id = str(dataset_path).replace("/", "_")
    train_dir = dataset_path / 'train.pkl'
    valid_dir = dataset_path / 'dev.pkl'
    test_dir = dataset_path / 'test.pkl'
    data_format = 'pkl'

    # read configuration from args.config
    dataset_config_path = dataset_path / 'config.json'
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)

    logger.info(f"Loaded dataset {dataset_id}: {dataset_path}")

    # fix the base_dir for the evaluation stage
    training_base_dir = training_output_config['base_config']['base_dir']
    prediction_base_dir = training_base_dir.replace("trained_models", "prediction_results")

    experiment_id = f"{training_output_config['base_config']['model_id']}_eval"

    # Transform the dict to match training configuration format
    config = {
        "pipeline_config_id": "runner_config",
        "data": {
            dataset_id: {
                "data_format": data_format,
                "train_dir": str(train_dir),
                "valid_dir": str(valid_dir),
                "test_dir": str(test_dir),
                "data_specs": {
                    "num_event_types": dataset_config["dim_process"],
                    "pad_token_id": dataset_config["dim_process"],
                    "padding_strategy" : 'do_not_pad',
                    "max_len": None
                }
            }
        },
        experiment_id: {
            "base_config": {
                "stage": "eval", # IMPORTANT
                "backend": training_output_config['base_config']['backend'],
                "dataset_id": dataset_id,
                "runner_id": training_output_config['base_config']['runner_id'],
                "model_id": training_output_config['base_config']['model_id'],
                "base_dir": prediction_base_dir,
            },
            "trainer_config": {
                "batch_size": batch_size,#training_output_config['trainer_config']['batch_size'],
                "max_epoch": training_output_config['trainer_config']['max_epoch'],
                "shuffle": training_output_config['trainer_config']['shuffle'],
                "optimizer": training_output_config['trainer_config']['optimizer'],
                "learning_rate": training_output_config['trainer_config']['learning_rate'],
                "valid_freq": training_output_config['trainer_config']['valid_freq'],
                "use_tfb": training_output_config['trainer_config']['use_tfb'],
                "metrics": training_output_config['trainer_config']['metrics'],
                "seed": training_output_config['trainer_config']['seed'],
                "gpu": gpu,#training_output_config['trainer_config']['gpu'],
            },
            "model_config": {
                "model_specs" : training_output_config['model_config']['model_specs'],
                "hidden_size": training_output_config['model_config']['hidden_size'],
                "num_layers": training_output_config['model_config']['num_layers'],
                "loss_integral_num_sample_per_step": training_output_config['model_config']['loss_integral_num_sample_per_step'],
                "use_ln": training_output_config['model_config']['use_ln'],
                "pretrained_model_dir": training_output_config['base_config']['specs']['saved_model_dir'],
                "thinning": {},
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            },
            "prediction_config" : prediction_config
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id)
    model_runner = TPPRunnerE2E(config)
    model_runner.run()