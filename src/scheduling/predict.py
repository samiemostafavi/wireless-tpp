import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml, pickle, json
import numpy as np

from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerScheduling
from wireless_tpp.utils import logger

def generate_predictions(args):

    # read configuration from args.config
    prediction_config_path = Path(args.config)
    with open(prediction_config_path, 'r') as f:
        prediction_config = json.load(f)
    prediction_config = prediction_config[args.configname]
    batch_size = prediction_config['batch_size']
    gpu = prediction_config['gpu']

    if prediction_config['dataset']:
        dataset_path = Path(prediction_config['dataset']) / "scheduling" / "datasets" / prediction_config['dataset_name']
    else:
        dataset_path = Path(args.source) / "scheduling" / "datasets" / args.name

    prediction_config['method'] = args.predict

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

    model_path = Path(args.source) / "scheduling" / "trained_models" / args.name / args.id
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
                "stage": "gen",
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
    model_runner = TPPRunnerScheduling(config)
    if args.predict == 'probabilistic':
        model_runner.run(probability_generation=True)
    else:
        model_runner.run()

def plot_predictions(args):

    # read configuration from args.config
    #dataset_config_path = Path(args.source) / "scheduling" / "datasets" / args.name / 'config.json'
    #with open(dataset_config_path, 'r') as f:
    #    dataset_config = json.load(f)
    dataset_config = None
    
    model_path = Path(args.source) / "scheduling" / "prediction_results" / args.name / args.id
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        generation_output_config = yaml.load(file, Loader=yaml.FullLoader)
    
    pkl_file = next(model_path.glob("*.pkl"))
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)

    model_id = generation_output_config['base_config']['model_id']
    if generation_output_config['prediction_config']['method'] == 'probabilistic':
        plot_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args)
    else:
        plot_sampling_predictions_1D(dataset_config, generation_output_config, data, model_path, args)


def plot_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args):

    segment_id = int(args.segment)
    num_event_types = generation_output_config['data_config']['data_specs']['num_event_types']
    #num_event_types_segment_only = (num_event_types-1)/2

    # we have 8 label attributes:
    # label_dtime, label_time, label_type, slot_seqs, len_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs
    # data['label'] dimensions: [num batches, 8 attributes , batch size, seq length]

    h_dtime, h_time, h_event_type, h_slot, h_len, h_mcs, h_mretx, h_rfailed, h_num_rbs = [],[],[],[],[],[],[],[],[]
    history_mcs_data = []
    for batch in data['label']:
        h_dtime.append(batch[0])
        h_time.append(batch[1])
        h_event_type.append(batch[2])
        h_slot.append(batch[3])
        h_len.append(batch[4])
        h_mcs.append(batch[5])
        h_mretx.append(batch[6])
        h_rfailed.append(batch[7])
        h_num_rbs.append(batch[8])

    ch_dtime = np.concatenate(h_dtime, axis=0)
    ch_time = np.concatenate(h_time, axis=0)
    ch_event_type = np.concatenate(h_event_type, axis=0)
    ch_slot = np.concatenate(h_slot, axis=0)
    ch_len = np.concatenate(h_len, axis=0)
    ch_mcs = np.concatenate(h_mcs, axis=0)
    ch_mretx = np.concatenate(h_mretx, axis=0)
    ch_rfailed = np.concatenate(h_rfailed, axis=0)
    ch_num_rbs = np.concatenate(h_num_rbs, axis=0)

    # data['pred'] dimensions: [num batches, 1 , batch size, num probability samples]
    p_dtime = []
    p_len_bytes = []
    for batch in data['pred']:
        p_dtime.append(batch[0])
        p_len_bytes.append(batch[1])
    cp_dtime = np.concatenate(p_dtime, axis=0)
    cp_len_bytes = np.concatenate(p_len_bytes, axis=0)

    # Here history data dimensions are: [total number of samples, seq length]
    # and prediction data dimensions are: [total number of samples, num probability samples]
    # total number of samples is the sum of all batch sizes

    # lets pick a sample and plot
    max_index = ch_dtime.shape[0]
    logger.info(f"Looking for segment id: {segment_id} in the history of size {max_index}")
    while True:
        ar_index = np.random.randint(0, max_index, size=1)[0]
        assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"
        if ch_event_type[ar_index,-1] == segment_id:
            break

    # [seq length]
    ch_dtime = ch_dtime[ar_index,:]
    ch_time = ch_time[ar_index,:]
    ch_event_type = ch_event_type[ar_index,:]
    ch_len = ch_len[ar_index,:]
    ch_mcs = ch_mcs[ar_index,:]
    ch_mretx = ch_mretx[ar_index,:]
    ch_rfailed = ch_rfailed[ar_index,:]
    ch_num_rbs = ch_num_rbs[ar_index,:]

    logger.info(f"Event types in the history plus the label: {ch_event_type}")

    # [num probability samples]
    cp_dtime = np.exp(cp_dtime[ar_index,:])
    # [num probability samples]
    cp_len_bytes = np.exp(cp_len_bytes[ar_index,:])


    # history packets time series
    packet_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mrtx_list = np.array([ch_mretx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_rrtx_list = np.array([ch_rfailed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])

    # history segments time series
    segment_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_type_list = np.array([ch_event_type[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_mrtx_list = np.array([ch_mretx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_rrtx_list = np.array([ch_rfailed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_dt_list = np.array([ch_dtime[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])


    # prediction dtime samples
    prediction_config = generation_output_config['prediction_config']
    sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
    sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
    num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
    dtime_samples = np.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime)


    # Create a subplot figure with 1 row
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Predictions"), specs=[[{"secondary_y": True}],[{"secondary_y": False}]])
    # Convert elements to strings

    # Combine the two lists
    #combined_list = 
    # Processed Events
    fig.add_trace(go.Scatter(x=packet_ts_list, y=np.ones(len(packet_ts_list)), mode='markers+text', name='Packet arrival (history)', marker=dict(symbol='square'), text=[f"{x},{y}" for x, y in zip(packet_mrtx_list, packet_rrtx_list)], textposition='top center', showlegend=False), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=packet_ts_list, y=np.ones(len(packet_ts_list)), mode='markers+text', name='Packet arrival (history)', marker=dict(symbol='square'), text=packet_len_list, textposition='bottom center'), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=segment_ts_list[:-1], y=np.ones(len(segment_ts_list[:-1])), mode='markers+text', name='Scheduling event (history)', marker=dict(symbol='circle'), text=[f"{x},{y}" for x, y in zip(segment_mrtx_list[:-2], segment_rrtx_list[:-2])], textposition='top center', showlegend=False), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=segment_ts_list[:-1], y=np.ones(len(segment_ts_list[:-1])), mode='markers+text', name='Scheduling event (history)', marker=dict(symbol='circle'), text=segment_len_list, textposition='bottom center'), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=segment_ts_list[-1:], y=np.ones(len(segment_ts_list[-1:])), mode='markers+text', name='Scheduling event (label)', marker=dict(symbol='circle'), text=[f"{x},{y}" for x, y in zip(segment_mrtx_list[-1:], segment_rrtx_list[-1:])], textposition='top center', showlegend=False), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=segment_ts_list[-1:], y=np.ones(len(segment_ts_list[-1:])), mode='markers+text', name='Scheduling event (label)', marker=dict(symbol='circle'), text=segment_len_list[-1:], textposition='bottom center'), row=1, col=1, secondary_y=True)

    fig.add_trace(
        go.Scatter(x=ch_time[-2]+dtime_samples, y=cp_dtime, mode='markers', name='predictions'),
        row=1, col=1,
        secondary_y=False
    )

    # prediction len samples
    sample_len_min = prediction_config['probability_generation']['sample_len_min']
    sample_len_max = prediction_config['probability_generation']['sample_len_max']
    num_steps_len = prediction_config['probability_generation']['num_steps_len']
    len_samples = np.linspace(sample_len_min, sample_len_max, num_steps_len)
    fig.add_trace(
        go.Scatter(x=len_samples, y=cp_len_bytes, mode='markers', name='Segment length prediction [bytes]'),
        row=2, col=1,
        secondary_y=False
    )

    fig.update_layout(
        title='Scheduling Predictor Validation',
        xaxis_title='Time [ms]',
        yaxis_title='Probability',
        legend_title='Legend',
        yaxis2=dict(showticklabels=False, title=None, overlaying='y', side='right', range=[0, 8])  # Set offset for the second y-axis
    )
    
    #fig.update_xaxes(matches='x')
    fig.write_html(model_path / "prob_delta_times.html")


def plot_sampling_predictions_1D(dataset_config, generation_output_config, data, model_path, args):

    segment_id = int(args.segment)
    num_event_types = generation_output_config['data_config']['data_specs']['num_event_types']
    #num_event_types_segment_only = (num_event_types-1)/2

    # we have 8 label attributes:
    # label_dtime, label_time, label_type, slot_seqs, len_seqs, mcs_seqs, mretx_seqs, rfailed_seqs, num_rbs_seqs
    # data['label'] dimensions: [num batches, 8 attributes , batch size, seq length]

    h_dtime, h_time, h_event_type, h_slot, h_len, h_mcs, h_mretx, h_rfailed, h_num_rbs = [],[],[],[],[],[],[],[],[]
    history_mcs_data = []
    for batch in data['label']:
        h_dtime.append(batch[0])
        h_time.append(batch[1])
        h_event_type.append(batch[2])
        h_slot.append(batch[3])
        h_len.append(batch[4])
        h_mcs.append(batch[5])
        h_mretx.append(batch[6])
        h_rfailed.append(batch[7])
        h_num_rbs.append(batch[8])

    ch_dtime = np.concatenate(h_dtime, axis=0)
    ch_time = np.concatenate(h_time, axis=0)
    ch_event_type = np.concatenate(h_event_type, axis=0)
    ch_slot = np.concatenate(h_slot, axis=0)
    ch_len = np.concatenate(h_len, axis=0)
    ch_mcs = np.concatenate(h_mcs, axis=0)
    ch_mretx = np.concatenate(h_mretx, axis=0)
    ch_rfailed = np.concatenate(h_rfailed, axis=0)
    ch_num_rbs = np.concatenate(h_num_rbs, axis=0)


    p_dtime = []
    p_num_rbs = []
    for batch in data['pred']:
        p_dtime.append(batch[0])
        p_num_rbs.append(batch[1])

    cp_dtime_samples = np.concatenate(p_dtime, axis=1)
    cp_num_rbs_samples = np.concatenate(p_num_rbs, axis=1)
    # cp_dtime_samples and cp_num_rbs_samples have the shape: [num gen samples, num samples in all the batches, 1]


    # Here history data dimensions are: [total number of samples, seq length]
    # and prediction data dimensions are: [total number of samples, num probability samples]
    # total number of samples is the sum of all batch sizes

    # lets pick a sample and plot
    max_index = cp_dtime_samples.shape[1]
    logger.info(f"Looking for segment id: {segment_id} in the history of size {max_index}")
    while True:
        ar_index = np.random.randint(0, max_index, size=1)[0]
        assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"
        if ch_event_type[ar_index,-1] == segment_id:
            break

    # [seq length]
    ch_dtime = ch_dtime[ar_index,:]
    ch_time = ch_time[ar_index,:]
    ch_event_type = ch_event_type[ar_index,:]
    ch_len = ch_len[ar_index,:]
    ch_mcs = ch_mcs[ar_index,:]
    ch_mretx = ch_mretx[ar_index,:]
    ch_rfailed = ch_rfailed[ar_index,:]
    ch_num_rbs = ch_num_rbs[ar_index,:]

    logger.info(f"Event types in the history plus the label: {ch_event_type}")

    # cp_dtime_samples and cp_num_rbs_samples have the shape: [num gen samples, num samples in all the batches, 1]
    cp_dtime = np.mean(cp_dtime_samples[:, ar_index, 0])
    # [1, 107]
    cp_num_rbs = np.mean(cp_num_rbs_samples[:, ar_index, 0])


    # history packets time series
    packet_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mrtx_list = np.array([ch_mretx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_rrtx_list = np.array([ch_rfailed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])

    # history segments time series
    segment_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_type_list = np.array([ch_event_type[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_mrtx_list = np.array([ch_mretx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_rrtx_list = np.array([ch_rfailed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_dt_list = np.array([ch_dtime[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])


    # Create a subplot figure with 1 row
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Predictions"))
    # Convert elements to strings

    # Combine the two lists
    #combined_list = 
    # Processed Events
    fig.add_trace(go.Scatter(x=packet_ts_list, y=np.ones(len(packet_ts_list)), mode='markers+text', name='Packet arrival (history)', marker=dict(symbol='square'), text=[f"{x},{y}" for x, y in zip(packet_mrtx_list, packet_rrtx_list)], textposition='top center', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=packet_ts_list, y=np.ones(len(packet_ts_list)), mode='markers+text', name='Packet arrival (history)', marker=dict(symbol='square'), text=packet_len_list, textposition='bottom center'), row=1, col=1)

    fig.add_trace(go.Scatter(x=segment_ts_list[:-1], y=np.ones(len(segment_ts_list[:-1])), mode='markers+text', name='Scheduling event (history)', marker=dict(symbol='circle'), text=[f"{x},{y}" for x, y in zip(segment_mrtx_list[:-2], segment_rrtx_list[:-2])], textposition='top center', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=segment_ts_list[:-1], y=np.ones(len(segment_ts_list[:-1])), mode='markers+text', name='Scheduling event (history)', marker=dict(symbol='circle'), text=segment_len_list, textposition='bottom center'), row=1, col=1)

    fig.add_trace(go.Scatter(x=segment_ts_list[-1:], y=np.ones(len(segment_ts_list[-1:])), mode='markers+text', name='Scheduling event (label)', marker=dict(symbol='circle'), text=[f"{x},{y}" for x, y in zip(segment_mrtx_list[-1:], segment_rrtx_list[-1:])], textposition='top center', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=segment_ts_list[-1:], y=np.ones(len(segment_ts_list[-1:])), mode='markers+text', name='Scheduling event (label)', marker=dict(symbol='circle'), text=segment_len_list[-1:], textposition='bottom center'), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=[ch_time[-2]+cp_dtime], y=[1], mode='markers+text', name='predictions', text=[cp_num_rbs], textposition='top center'),
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