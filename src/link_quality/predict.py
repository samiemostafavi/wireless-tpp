import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml, pickle, json
import numpy as np

from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerLinkQuality
from wireless_tpp.utils import logger

def generate_predictions(args):

    # read configuration from args.config
    dataset_config_path = Path(args.source) / "link_quality" / "datasets" / args.name / 'config.json'
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)

    # read configuration from args.config
    prediction_config_path = Path(args.config)
    with open(prediction_config_path, 'r') as f:
        prediction_config = json.load(f)
    prediction_config = prediction_config[args.configname]
    batch_size = prediction_config['batch_size']
    gpu = prediction_config['gpu']
    prediction_config['method'] = args.predict

    model_path = Path(args.source) / "link_quality" / "trained_models" / args.name / args.id
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    # fix the base_dir for the generation stage
    training_base_dir = training_output_config['base_config']['base_dir']
    prediction_base_dir = training_base_dir.replace("trained_models", "prediction_results")

    experiment_id = f"{training_output_config['base_config']['model_id']}_gen"
    # Transform the dict to match training configuration format
    config = {
        "pipeline_config_id": "runner_config",
        "data": {
            training_output_config['base_config']['dataset_id']: {
                "data_format": training_output_config['data_config']['data_format'],
                "train_dir": training_output_config['data_config']['train_dir'],
                "valid_dir": training_output_config['data_config']['valid_dir'],
                "test_dir": training_output_config['data_config']['test_dir'],
                "data_specs": {
                    "num_event_types": training_output_config['data_config']['data_specs']['num_event_types'],
                    "pad_token_id": training_output_config['data_config']['data_specs']['pad_token_id'],
                    "padding_side": training_output_config['data_config']['data_specs']['padding_side'],
                    "truncation_side": training_output_config['data_config']['data_specs']['truncation_side'],
                    "padding_strategy" : training_output_config['data_config']['data_specs']['padding_strategy'],
                    "max_len": training_output_config['data_config']['data_specs']['max_len'],
                    "includes_mcs" : training_output_config['data_config']['data_specs']['includes_mcs'],
                    "num_event_types_no_mcs": training_output_config['data_config']['data_specs']['num_event_types_no_mcs'],
                    "min_mcs": training_output_config['data_config']['data_specs']['min_mcs'],
                    "mcs_events": training_output_config['data_config']['data_specs']['mcs_events']
                }
            }
        },
        experiment_id: {
            "base_config": {
                "stage": "gen",
                "backend": training_output_config['base_config']['backend'],
                "dataset_id": training_output_config['base_config']['dataset_id'],
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
                "thinning": prediction_config['thinning'],
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            },
            "prediction_config" : prediction_config
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id)
    model_runner = TPPRunnerLinkQuality(config)
    if args.predict == 'probabilistic':
        model_runner.run(probability_generation=True)
    else:
        model_runner.run()

def plot_predictions(args):

    # read configuration from args.config
    dataset_config_path = Path(args.source) / "link_quality" / "datasets" / args.name / 'config.json'
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)
    
    model_path = Path(args.source) / "link_quality" / "prediction_results" / args.name / args.id
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        generation_output_config = yaml.load(file, Loader=yaml.FullLoader)
    
    pkl_file = next(model_path.glob("*.pkl"))
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)

    mcs_eval_interval_ms = 100
    model_id = generation_output_config['base_config']['model_id']
    if generation_output_config['prediction_config']['method'] == 'probabilistic':
        plot_mcs_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args)
    elif generation_output_config['prediction_config']['method'] == 'sampling':
        plot_mcs_sampling_predictions_1D(dataset_config, generation_output_config, data, model_path, mcs_eval_interval_ms, args)


def plot_mcs_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args):

    # history data
    h_dtime, h_time, h_event_type, h_mcs, h_mretx, h_rfailed, h_num_rbs = [],[],[],[],[],[],[]
    history_mcs_data = []
    for batch in data['label']:
        h_dtime.append(batch[0])
        h_time.append(batch[1])
        h_event_type.append(batch[2])
        h_mcs.append(batch[3])
        h_mretx.append(batch[4])
        h_rfailed.append(batch[5])
        h_num_rbs.append(batch[6])

    ch_dtime = np.concatenate(h_dtime, axis=0)
    ch_time = np.concatenate(h_time, axis=0)
    ch_event_type = np.concatenate(h_event_type, axis=0)
    ch_mcs = np.concatenate(h_mcs, axis=0)
    ch_mretx = np.concatenate(h_mretx, axis=0)
    ch_rfailed = np.concatenate(h_rfailed, axis=0)
    ch_num_rbs = np.concatenate(h_num_rbs, axis=0)

    # data['pred'] dimensions: [num batches, 1 , batch size, num probability samples]
    p_mcs = []
    for batch in data['pred']:
        p_mcs.append(batch[0])
    cp_mcs = np.concatenate(p_mcs, axis=0)

    # Here history data dimensions are: [total number of samples, seq length]
    # and prediction data dimensions are: [total number of samples, num probability samples]
    # total number of samples is the sum of all batch sizes

    # lets pick a sample and plot
    max_index = ch_dtime.shape[0]

    ar_index = np.random.randint(0, max_index, size=1)[0]
    assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"

    # [seq length]
    ch_dtime = ch_dtime[ar_index,:]
    ch_time = ch_time[ar_index,:]
    ch_event_type = ch_event_type[ar_index,:]
    ch_mcs = ch_mcs[ar_index,:]
    ch_mretx = ch_mretx[ar_index,:]
    ch_rfailed = ch_rfailed[ar_index,:]
    ch_num_rbs = ch_num_rbs[ar_index,:]

    #logger.info(f"Event types in the history plus the label: {ch_event_type}")

    # [num probability samples]
    cp_mcs = np.exp(cp_mcs[ar_index,:])

    # history mcs event time series
    mcsevent_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 1])
    mcsevent_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 1])

    # history segments time series
    segment_mrtx_list = np.array([ch_mretx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_rfailed_list = np.array([ch_rfailed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_num_rbs_list = np.array([ch_num_rbs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])

    # Create a subplot figure with 1 row
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Predictions"))

    # history mcs events
    fig.add_trace(go.Scatter(x=mcsevent_ts_list[:-1], y=np.ones(len(mcsevent_ts_list)), mode='markers+text', name='MCS event (history)', marker=dict(symbol='square'), text=[f"{x}" for x in mcsevent_mcs_list[:-1]], textposition='top center', showlegend=True), row=1, col=1)

    # history block events
    fig.add_trace(go.Scatter(x=segment_ts_list, y=np.ones(len(segment_ts_list)), mode='markers+text', name='Block event (history)', marker=dict(symbol='circle'), text=[f"{x},{y}" for x, y in zip(segment_mrtx_list, segment_rfailed_list)], textposition='top center', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=segment_ts_list, y=np.ones(len(segment_ts_list)), mode='markers+text', name='Block event (history)', marker=dict(symbol='circle'), text=segment_num_rbs_list, textposition='bottom center'), row=1, col=1)

    # label mcs event
    fig.add_trace(go.Scatter(x=mcsevent_ts_list[-1:], y=np.ones(len(mcsevent_ts_list)), mode='markers+text', name='MCS event (label)', marker=dict(symbol='square'), text=[f"{x}" for x in mcsevent_mcs_list[-1:]], textposition='top center', showlegend=True), row=1, col=1)

    # prediction probabilities
    fig.add_trace(go.Scatter(x=np.arange(30), y=cp_mcs[0,:], mode='markers', name='Segment length prediction [bytes]'), row=2, col=1)

    fig.update_layout(
        title='MCS Predictor Validation',
        legend_title='Legend',
    )

    # Write the plot to an HTML file
    fig.write_html(model_path / "pred_probabilities_mcs.html")


def transform_list(input_list, max_period):
    # Initialize an empty list to store the transformed values
    transformed_list = []
    
    # Keep track of period-based offset for each segment
    offset = 0
    previous_value = None

    for i, value in enumerate(input_list):
        # Check if there is a decrease or reset to a lower number (assumed new period start)
        if previous_value is not None and value < previous_value:
            offset += max_period  # Decrease offset by max_period
        
        # Calculate and append the new value
        transformed_value = value + offset
        transformed_list.append(transformed_value)
        
        # Update the previous value
        previous_value = value
    
    return transformed_list

def plot_mcs_sampling_predictions_1D(dataset_config, generation_output_config, data, model_path, mcs_eval_interval_ms, args):

    # history data
    h_dtime, h_time, h_event_type, h_mcs, h_mretx, h_rfailed, h_num_rbs = [],[],[],[],[],[],[]
    history_mcs_data = []
    for batch in data['label']:
        h_dtime.append(batch[0])
        h_time.append(batch[1])
        h_event_type.append(batch[2])
        h_mcs.append(batch[3])
        h_mretx.append(batch[4])
        h_rfailed.append(batch[5])
        h_num_rbs.append(batch[6])

    ch_dtime = np.concatenate(h_dtime, axis=0)
    ch_time = np.concatenate(h_time, axis=0)
    ch_event_type = np.concatenate(h_event_type, axis=0)
    ch_mcs = np.concatenate(h_mcs, axis=0)
    ch_mretx = np.concatenate(h_mretx, axis=0)
    ch_rfailed = np.concatenate(h_rfailed, axis=0)
    ch_num_rbs = np.concatenate(h_num_rbs, axis=0)

    # data['pred'] dimensions: [num batches, 1 , batch size, num probability samples]
    p_mcs = []
    for batch in data['pred']:
        p_mcs.append(batch[0])
    cp_mcs = np.concatenate(p_mcs, axis=1)

    # Here history data dimensions are: [total number of samples, seq length]
    # and prediction data dimensions are: [total number of samples, num probability samples]
    # total number of samples is the sum of all batch sizes

    # lets pick a sample and plot
    max_index = ch_dtime.shape[1]

    ar_index = np.random.randint(0, max_index, size=1)[0]
    assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"

    # [seq length]
    ch_dtime = ch_dtime[ar_index,:]
    ch_time = ch_time[ar_index,:]
    ch_event_type = ch_event_type[ar_index,:]
    ch_mcs = ch_mcs[ar_index,:]
    ch_mretx = ch_mretx[ar_index,:]
    ch_rfailed = ch_rfailed[ar_index,:]
    ch_num_rbs = ch_num_rbs[ar_index,:]

    #logger.info(f"Event types in the history plus the label: {ch_event_type}")

    # [num probability samples]
    cp_mcs = np.mean(cp_mcs[:,ar_index,0])

    # history mcs event time series
    mcsevent_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 1])
    mcsevent_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 1])

    # history segments time series
    segment_mrtx_list = np.array([ch_mretx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_rfailed_list = np.array([ch_rfailed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_num_rbs_list = np.array([ch_num_rbs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    segment_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])

    # Create a subplot figure with 1 row
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Predictions"))

    # history mcs events
    fig.add_trace(go.Scatter(x=mcsevent_ts_list[:-1], y=np.ones(len(mcsevent_ts_list)), mode='markers+text', name='MCS event (history)', marker=dict(symbol='square'), text=[f"{x}" for x in mcsevent_mcs_list[:-1]], textposition='top center', showlegend=True), row=1, col=1)

    # history block events
    fig.add_trace(go.Scatter(x=segment_ts_list, y=np.ones(len(segment_ts_list)), mode='markers+text', name='Block event (history)', marker=dict(symbol='circle'), text=[f"{x},{y}" for x, y in zip(segment_mrtx_list, segment_rfailed_list)], textposition='top center', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=segment_ts_list, y=np.ones(len(segment_ts_list)), mode='markers+text', name='Block event (history)', marker=dict(symbol='circle'), text=segment_num_rbs_list, textposition='bottom center'), row=1, col=1)

    # label mcs event
    fig.add_trace(go.Scatter(x=mcsevent_ts_list[-1:], y=np.ones(len(mcsevent_ts_list)), mode='markers+text', name='MCS event (label)', marker=dict(symbol='square'), text=[f"{x}" for x in mcsevent_mcs_list[-1:]], textposition='top center', showlegend=True), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=[ch_time[-2]+mcs_eval_interval_ms], y=[1], mode='markers+text', name='predictions', text=[cp_mcs], textposition='top center'),
        row=1, col=1
    )

    fig.update_layout(
        title='Scheduling Predictor Validation',
        xaxis_title='Time [ms]',
        yaxis_title='Value',
        legend_title='Legend'
    )
    
    #fig.update_xaxes(matches='x')
    fig.write_html(model_path / "pred_sample_dtimes.html")