from pathlib import Path
import yaml, pickle, json
import numpy as np

from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerE2E
from wireless_tpp.utils import logger
import argparse
            

def generate_predictions(args, prediction_base_dir):

    # read configuration from args.config
    prediction_config_path = Path(args.config)
    with open(prediction_config_path, 'r') as f:
        prediction_config = json.load(f)
    prediction_config = prediction_config[args.configname]
    batch_size = prediction_config['batch_size']
    gpu = prediction_config['gpu']

    prediction_config['method'] = args.predict

    if args.id:
        model_path = Path(args.source) / "e2e" / "trained_models" / args.name / args.id
    else:
        # if no id is passed, take the first folder in the trained_models directory
        tranied_models_path = Path(args.source) / "e2e" / "trained_models" / args.name
        model_path = next(tranied_models_path.iterdir())

    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    # fix the base_dir for the generation stage
    training_base_dir = training_output_config['base_config']['base_dir']

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

    experiment_id = f"{training_output_config['base_config']['model_id']}_gen"

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
                "stage": "gen", # IMPORTANT
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

    predictions = model_runner.run(return_predictions=True)
    return predictions



def main():
    parser = argparse.ArgumentParser(description="Validation plot parser")

    parser.add_argument("-s", "--source", help="Specify the source directory")
    parser.add_argument("-f", "--fast", action="store_true", help="Specify if in plot_link_data, only priliminary data should be plotted")
    parser.add_argument("-c", "--config", help="Specify the configuration file")
    parser.add_argument("-g", "--configname", help="Specify the configuration name in the configuration file")
    parser.add_argument("-n", "--name", help="Specify the name of the dataset")
    parser.add_argument("-i", "--id", help="Specify the training id")
    parser.add_argument("-p", "--predict", choices=["probabilistic","sampling"],help="Specify the prediction method")
    args, remaining_args = parser.parse_known_args()

    model_name = args.name

    predictions_dir = "tmp"
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)

    # this function takes the test dataset (50k) and generates predictions for each sequence in there
    predictions = generate_predictions(args, predictions_dir)

    """
    predictions is a list of following dict
    {
        'src_seqs_len': self.model_wrapper.model.src_seq_len,
        'tgt_seqs_len': self.model_wrapper.model.tgt_seq_len,
        'features' : {
            'slot_seqs': slot_seqs[seq_num],
            'len_seqs': len_seqs_transformed[seq_num],
            'mcs_seqs': mcs_seqs[seq_num],
            'mretx_seqs': mretx_seqs[seq_num],
            'rfailed_seqs': rfailed_seqs[seq_num],
            'interarrival_time_seqs': interarrival_time_seqs_transformed[seq_num]
        },
        'labels' : dtime_seqs_transformed[seq_num],
        'predictions' : {
            'pred_mean': pred_mean[seq_num],
            'pred_var': pred_var[seq_num],
            'pred_q5a': pred_q5a[seq_num],
            'pred_q5b': pred_q5b[seq_num],
            'pred_q7a': pred_q7a[seq_num],
            'pred_q7b': pred_q7b[seq_num],
            'pred_q9a': pred_q9a[seq_num],
            'pred_q9b': pred_q9b[seq_num],
            'pred_q99a': pred_q99a[seq_num],
            'pred_q99b': pred_q99b[seq_num]
        }
    }
    """
    # plot the sequence of delay values from the source sequence:
    # source sequence: [-src_seq_len-tgt_seq_len:-tgt_seq_len]
    # target sequence: [-tgt_seq_len:]
    # then for the target sequence, print the prediction probability distribution using the coverage values (q5a, q5b, q7a, q7b, q9a, q9b, q99a, q99b)

    import matplotlib.pyplot as plt

    # Ensure IEEE-compliant font and style
    # IEEE Transactions format settings
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,  # Font size as per IEEE standards
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "lines.linewidth": 1,
        "axes.linewidth": 0.8
    })

    figures_path = Path("./figures_s61-64_valid")
    # Create the figures directory if it doesn't exist
    figures_path.mkdir(parents=True, exist_ok=True)

    # here we could do better by selecting a random element instead of the first
    prediction = predictions[0]
    
    src_seqs_len = prediction['src_seqs_len']
    tgt_seqs_len = prediction['tgt_seqs_len']
    features = prediction['features']
    labels = prediction['labels']
    pred_mean = prediction['predictions']['pred_mean']
    pred_var = prediction['predictions']['pred_var']
    pred_q5a = prediction['predictions']['pred_q5a']
    pred_q5b = prediction['predictions']['pred_q5b']
    pred_q7a = prediction['predictions']['pred_q7a']
    pred_q7b = prediction['predictions']['pred_q7b']
    pred_q9a = prediction['predictions']['pred_q9a']
    pred_q9b = prediction['predictions']['pred_q9b']
    pred_q99a = prediction['predictions']['pred_q99a']
    pred_q99b = prediction['predictions']['pred_q99b']

    # Plot the source sequence
    # Create a new figure for combined plots
    plt.figure(figsize=(4, 2.5))
    plt.plot(range(-src_seqs_len, 0), labels[-src_seqs_len-tgt_seqs_len:-tgt_seqs_len], label='Delay History')

    # Plot the target sequence
    plt.plot(range(0, tgt_seqs_len), labels[-tgt_seqs_len:], label='Ground Truth')

    # Plot the prediction probability distribution
    plt.fill_between(range(0, tgt_seqs_len), pred_q5a, pred_q5b, color='blue', alpha=0.1)
    plt.fill_between(range(0, tgt_seqs_len), pred_q7a, pred_q7b, color='green', alpha=0.1)
    plt.fill_between(range(0, tgt_seqs_len), pred_q9a, pred_q9b, color='red', alpha=0.1)
    plt.fill_between(range(0, tgt_seqs_len), pred_q99a, pred_q99b, color='purple', alpha=0.1)

    plt.plot(range(0, tgt_seqs_len), pred_mean, label='Prediction Dist. Mean', linestyle='--')

    plt.xlabel('Time Step')
    plt.ylabel('Packet Delay [ms]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_path / f"{model_name}_{src_seqs_len}_{tgt_seqs_len}.pdf")
    plt.close()


if __name__ == "__main__":
    main()


# Figure one:
# python paper_plot_valid.py -s data/s61-64_results -c config/prediction_config.json -g validate -n 10k_transformer_100_EXC
# Figure two:
# python paper_plot_valid.py -s data/s61-64_results -c config/prediction_config.json -g validate -n 10k_mlp_100_EXC