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

    model_id = generation_output_config['base_config']['model_id']
    if model_id == 'IntensityFreeLinkQuality' and generation_output_config['prediction_config']['method'] == 'probabilistic':
        plot_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args)
    elif model_id == 'THPLinkQuality' and generation_output_config['prediction_config']['method'] == 'probabilistic':
        plot_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args)
    elif model_id == 'THPLinkQuality' and generation_output_config['prediction_config']['method'] == 'sampling':
        plot_sampling_predictions_1D(dataset_config, generation_output_config, data, model_path, args)


def plot_sampling_predictions_1D(dataset_config, generation_output_config, data, model_path, args):

    includes_mcs = False
    if generation_output_config['model_config']['model_specs']['includes_mcs']:
        includes_mcs = True

    # plot history
    history_dtime_data = []
    history_event_type_data = []
    history_mcs_data = []
    for batch in data['label']:
        history_dtime_data.append(batch[0])
        history_event_type_data.append(batch[1])
        if includes_mcs:
            history_mcs_data.append(batch[2])

    concatenated_history_dtime = np.concatenate(history_dtime_data, axis=0)
    concatenated_history_event_type = np.concatenate(history_event_type_data, axis=0)
    if includes_mcs:
        concatenated_history_mcs = np.concatenate(history_mcs_data, axis=0)

    dtime_data = []
    event_type_data = []
    for batch in data['pred']:
        dtime_data.append(batch[0])
        event_type_data.append(batch[1])
    concatenated_label_dtime = np.concatenate(dtime_data, axis=0)
    concatenated_label_event_type = np.concatenate(event_type_data, axis=0)

    max_index = concatenated_label_dtime.shape[0]
    ar_index = np.random.randint(0, max_index, size=1)[0]
    assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"
    dtime_pred = concatenated_label_dtime[ar_index,0]
    event_type_pred = concatenated_label_event_type[ar_index,0]

    history_dtime = concatenated_history_dtime[ar_index,:]
    history_time = np.cumsum(history_dtime)
    history_event_type = concatenated_history_event_type[ar_index,:]
    if includes_mcs:
        history_mcs = concatenated_history_mcs[ar_index,:]


    # Create a subplot with 2 rows and 1 column
    if includes_mcs:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Predictions", "History MCS"))
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Predictions"))


    # Add the scatter plot for predictions to the first row
    fig.add_trace(
        go.Scatter(x=history_time[:-1], y=history_event_type[:-1], mode='markers', name='labels'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=[history_time[-1]], y=[history_event_type[-1]], mode='markers', name='labels'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=[history_time[-1]+dtime_pred], y=[event_type_pred], mode='markers', name='predictions'),
        row=1, col=1
    )

    if includes_mcs:
        # Add the scatter plot for history MCS to the second row
        fig.add_trace(
            go.Scatter(x=np.arange(len(history_mcs)), y=history_mcs, mode='lines+markers', name='History MCS'),
            row=3, col=1
        )

    # Update layout for better aesthetics
    fig.update_layout(
        title='Predictions and History',
        xaxis_title='Time',
        yaxis_title='Probability'
    )

    # Write the plot to an HTML file
    fig.write_html(model_path / "pred_sample_dtimes.html")

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

def plot_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args):

    includes_mcs = False
    if generation_output_config['model_config']['model_specs']['includes_mcs']:
        includes_mcs = True

    mcs_events = False
    if generation_output_config['data_config']['data_specs']['mcs_events']:
        mcs_events = True
        num_event_types_no_mcs = generation_output_config['data_config']['data_specs']['num_event_types_no_mcs']
        min_mcs = generation_output_config['data_config']['data_specs']['min_mcs']

    # plot history
    history_dtime_data = []
    history_time_data = []
    history_event_type_data = []
    history_mcs_data = []
    for batch in data['label']:
        history_dtime_data.append(batch[0])
        history_time_data.append(batch[1])
        history_event_type_data.append(batch[2])
        if includes_mcs:
            history_mcs_data.append(batch[3])


    concatenated_history_dtime = np.concatenate(history_dtime_data, axis=0)
    concatenated_history_time = np.concatenate(history_time_data, axis=0)
    concatenated_history_event_type = np.concatenate(history_event_type_data, axis=0)
    if includes_mcs:
        concatenated_history_mcs = np.concatenate(history_mcs_data, axis=0)

    dtime_data = []
    event_type_data = []
    for batch in data['pred']:
        dtime_data.append(batch[0])
        event_type_data.append(batch[1])
    concatenated_label_dtime = np.concatenate(dtime_data, axis=0)
    concatenated_label_event_type = np.concatenate(event_type_data, axis=0)

    max_index = concatenated_label_dtime.shape[0]
    ar_index = np.random.randint(0, max_index, size=1)[0]
    assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"
    dtime_logprob_pred = concatenated_label_dtime[ar_index,:]
    dtime_prob_pred = np.exp(dtime_logprob_pred)

    history_dtime = concatenated_history_dtime[ar_index,:]
    history_time = concatenated_history_time[ar_index,:]
    history_event_type = concatenated_history_event_type[ar_index,:]
    if includes_mcs:
        history_mcs = concatenated_history_mcs[ar_index,:]

    # dtime samples
    prediction_config = generation_output_config['prediction_config']
    sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
    sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
    num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
    dtime_samples = np.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime)
    
    # Create a subplot with 2 rows and 1 column
    if includes_mcs:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Predictions", "History MCS"))
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Predictions"), specs=[[{"secondary_y": True}]])

    history_time = transform_list(history_time, 1024*10.0) # 1024 (max_num_frames) * 10ms (20 slots each 0.5 ms) is the max period

    # Add the scatter plot for predictions to the first row
    if mcs_events:
        history_event_type = np.where(
            np.array(history_event_type) >= num_event_types_no_mcs,
            np.array(history_event_type) + min_mcs - num_event_types_no_mcs,
            np.array(history_event_type)
        )

    fig.add_trace(
        go.Scatter(
            x=history_time[:-1], 
            y=np.where(np.array(history_event_type[:-1]) >= num_event_types_no_mcs, num_event_types_no_mcs, np.array(history_event_type[:-1])), 
            mode='markers+text', 
            text=np.where(np.array(history_event_type[:-1]) >= num_event_types_no_mcs, np.array(history_event_type[:-1]), ''),
            textposition='bottom center',
            name='History events'
        ),
        row=1, col=1,
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=[history_time[-1]], 
            y=np.where(np.array([history_event_type[-1]]) >= num_event_types_no_mcs, num_event_types_no_mcs, np.array([history_event_type[-1]])), 
            mode='markers+text', 
            text=np.where(np.array([history_event_type[-1]]) >= num_event_types_no_mcs, np.array([history_event_type[-1]]), ''),
            textposition='bottom center',
            name='Label event'
        ),
        row=1, col=1,
        secondary_y=False
    )

    # Add the scatter plot for predictions to the first row
    #fig.add_trace(
    #    go.Scatter(x=history_time, y=np.ones(len(history_dtime)), mode='markers', name='labels'),
    #    row=2, col=1
    #)

    fig.add_trace(
        go.Scatter(x=history_time[-2]+dtime_samples, y=dtime_prob_pred, mode='markers', name='predictions'),
        row=1, col=1,
        secondary_y=True
    )  

    # Add the scatter plot for predictions to the first row
    #fig.add_trace(
    #    go.Scatter(x=dtime_samples, y=dtime_prob_pred, mode='markers', name='Predictions'),
    #    row=1, col=1
    #)

    if includes_mcs:
        # Add the scatter plot for history MCS to the second row
        fig.add_trace(
            go.Scatter(x=np.arange(len(history_mcs)), y=history_mcs, mode='lines+markers', name='History MCS'),
            row=2, col=1
        )

    # Update layout for better aesthetics
    fig.update_layout(
        title="Predictions and History",
        xaxis_title="Time [ms]",
        yaxis_title="History Events [event type]",
        yaxis2_title="Prediction Probability"
    )

    # Write the plot to an HTML file
    fig.write_html(model_path / "prob_delta_times.html")

    # (951, 999, 15)
    event_types_length = concatenated_label_event_type.shape[2] # event_types length
    event_types_probs = np.exp(concatenated_label_event_type[ar_index,0,:]) #FIXME why index 0?

    # Define class labels
    class_labels = [f"Event type {i}" for i in range(event_types_length)]

    # Create a figure
    fig = go.Figure()

    # Add traces for each set of probabilities
    fig.add_trace(
        go.Bar(
            x=class_labels,
            y=event_types_probs,
            opacity=0.7
        )
    )
    # Update layout for better aesthetics
    fig.update_layout(
        title="Probability Distribution Across Classes",
        xaxis_title="Class Number",
        yaxis_title="Probability",
        barmode='group',  # Options: 'group', 'overlay', 'stack'
        template='plotly_white',
        legend_title="Probability Sets",
        xaxis_tickangle=-45
    )
    fig.write_html(model_path / "prob_event_types.html")