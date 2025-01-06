import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml, pickle, json, copy
import numpy as np

from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerPacketArrival, TPPRunnerLinkQuality, TPPRunnerScheduling
from wireless_tpp.utils import logger

from edaf.core.uplink.analyze_packet import ULPacketAnalyzer
from edaf.core.uplink.analyze_scheduling import ULSchedulingAnalyzer
from edaf.core.uplink.analyze_channel import ULChannelAnalyzer

from src.link_quality import extract_link_quality_events
from src.packet_arrival import extract_packet_arrival_events
from src.scheduling import extract_scheduling_events
from .sample_e2e import sample_based_e2e_prediction


def create_arrival_exp_config(args, arrival_conf, batch_size, gpu, prediction_base_dir):
    model_path = Path(args.source) / "packet_arrival" / "trained_models" / arrival_conf['trained_model_name'] / arrival_conf['trained_model_id']
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    # fix the base_dir for the generation stage
    training_base_dir = training_output_config['base_config']['base_dir']
    #prediction_base_dir = training_base_dir.replace("trained_models", "prediction_results")

    experiment_id = f"{training_output_config['base_config']['model_id']}_gen"
    # Transform the dict to match training configuration format
                
    config = {
        "pipeline_config_id": "runner_config",
        "data": {},
        experiment_id: {
            "base_config": {
                "stage": "gen",
                "backend": training_output_config['base_config']['backend'],
                "dataset_id": None,
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
                "thinning": arrival_conf['thinning'] if 'thinning' in arrival_conf else {},
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            },
            "prediction_config" : arrival_conf
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id, no_dataset=True)
    return config

def create_link_quality_exp_config(args, link_quality_conf, batch_size, gpu, prediction_base_dir):
    model_path = Path(args.source) / "link_quality" / "trained_models" / link_quality_conf['trained_model_name'] / link_quality_conf['trained_model_id']
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    # fix the base_dir for the generation stage
    training_base_dir = training_output_config['base_config']['base_dir']
    #prediction_base_dir = training_base_dir.replace("trained_models", "prediction_results")

    experiment_id = f"{training_output_config['base_config']['model_id']}_gen"
    # Transform the dict to match training configuration format
    config = {
        "pipeline_config_id": "runner_config",
        "data": {},
        experiment_id: {
            "base_config": {
                "stage": "gen",
                "backend": training_output_config['base_config']['backend'],
                "dataset_id": None,
                "runner_id": training_output_config['base_config']['runner_id'],
                "model_id": training_output_config['base_config']['model_id'],
                "base_dir": prediction_base_dir,
            },
            "trainer_config": {
                "batch_size": batch_size,
                "max_epoch": training_output_config['trainer_config']['max_epoch'],
                "shuffle": training_output_config['trainer_config']['shuffle'],
                "optimizer": training_output_config['trainer_config']['optimizer'],
                "learning_rate": training_output_config['trainer_config']['learning_rate'],
                "valid_freq": training_output_config['trainer_config']['valid_freq'],
                "use_tfb": training_output_config['trainer_config']['use_tfb'],
                "metrics": training_output_config['trainer_config']['metrics'],
                "seed": training_output_config['trainer_config']['seed'],
                "gpu": gpu,
            },
            "model_config": {
                "model_specs" : training_output_config['model_config']['model_specs'],
                "hidden_size": training_output_config['model_config']['hidden_size'],
                "num_layers": training_output_config['model_config']['num_layers'],
                "loss_integral_num_sample_per_step": training_output_config['model_config']['loss_integral_num_sample_per_step'],
                "use_ln": training_output_config['model_config']['use_ln'],
                "pretrained_model_dir": training_output_config['base_config']['specs']['saved_model_dir'],
                "thinning": link_quality_conf['thinning'] if 'thinning' in link_quality_conf else {},
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            },
            "prediction_config" : link_quality_conf
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id, no_dataset=True)
    return config

def create_scheduling_exp_config(args, scheduling_conf, batch_size, gpu, prediction_base_dir):
    model_path = Path(args.source) / "scheduling" / "trained_models" / scheduling_conf['trained_model_name'] / scheduling_conf['trained_model_id']
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    # fix the base_dir for the generation stage
    training_base_dir = training_output_config['base_config']['base_dir']
    #prediction_base_dir = training_base_dir.replace("trained_models", "prediction_results")

    experiment_id = f"{training_output_config['base_config']['model_id']}_gen"
    # Transform the dict to match training configuration format
    config = {
        "pipeline_config_id": "runner_config",
        "data": {},
        experiment_id: {
            "base_config": {
                "stage": "gen",
                "backend": training_output_config['base_config']['backend'],
                "dataset_id": None,
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
                "thinning": scheduling_conf['thinning'] if 'thinning' in scheduling_conf else {},
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            },
            "prediction_config" : scheduling_conf
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id, no_dataset=True)
    return config


def generate_predictions(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    # select the source configuration
    e2e_config = dataset_config[args.configname]
    gpu = e2e_config['gpu']
    batch_size = e2e_config['batch_size']
    dataset_name = e2e_config['dataset_name']
    arrival_pred_conf = e2e_config['arrival']
    sched_pred_conf = e2e_config['scheduling']
    mcs_pred_conf = e2e_config['mcs']
    retx_pred_conf = e2e_config['retx']

    # read experiment configuration
    folder_addr = Path(args.source)
    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    # read the dataset
    with open(folder_addr / 'e2e' / 'datasets' / dataset_name / 'config.json', 'r') as f:
        dataset_config = json.load(f)
    with open(folder_addr / 'e2e' / 'datasets' / dataset_name / 'test.pkl', 'rb') as f:
        dataset = pickle.load(f)

    logger.info(f"Loaded the dataset, Number of total entries: {len(dataset)}")

    # prepare the results folder
    results_folder_addr = folder_addr / 'e2e' / 'prediction_results' /  args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)

    # load arrival model configuration
    arrival_exp_config = create_arrival_exp_config(args, arrival_pred_conf, batch_size, gpu, results_folder_addr)
    arrival_runner = TPPRunnerPacketArrival(
        runner_config=arrival_exp_config,
        unique_model_dir=False
    )

    # mcs model configuration
    mcs_exp_config = create_link_quality_exp_config(args, mcs_pred_conf, batch_size, gpu, results_folder_addr)
    mcs_runner = TPPRunnerLinkQuality(
        runner_config=mcs_exp_config,
        unique_model_dir=False
    )

    # retx model configuration
    retx_exp_config = create_link_quality_exp_config(args, retx_pred_conf, batch_size, gpu, results_folder_addr)
    retx_runner = TPPRunnerLinkQuality(
        runner_config=retx_exp_config,
        unique_model_dir=False
    )

    # load scheduling model configuration
    scheduling_exp_config = create_scheduling_exp_config(args, sched_pred_conf, batch_size, gpu, results_folder_addr)
    sched_runner = TPPRunnerScheduling(
        runner_config=scheduling_exp_config,
        unique_model_dir=False
    )

    # run e2e delay prediction
    # for each entry in dataset, we run one prediction
    for entry in dataset:
        predicted_packet_transmissions = sample_based_e2e_prediction(
            data = entry,
            arrival_runner = arrival_runner, 
            mcs_runner = mcs_runner, 
            retx_runner = retx_runner,
            sched_runner = sched_runner, 
            exp_config = exp_config,
            num_packets = 2,
            mcs_eval_interval_ms = 100,
            filter_successful_attempts_for_mcs = True,
            mcs_dimension_limit = 10,
            history_dimension_limit = 10,
            exclude_link_quality = False,
            max_num_segments = 5
        )
        print(entry['label'])
        print('-')
        print(predicted_packet_transmissions[0][0])
        print('-')
        print(predicted_packet_transmissions[1][0])
        input()

def plot_data(args):
    # read configuration from args.config
    with open(args.config, 'r') as f:
        e2e_config = json.load(f)
    
    time_masks = e2e_config['time_masks']
    gpu = e2e_config['gpu']

    # read experiment configuration
    folder_addr = Path(args.source)
    # find all .db files in the folder
    db_files = list(folder_addr.glob("*.db"))
    if not db_files:
        logger.error("No database files found in the specified folder.")
        return
    result_database_files = [str(db_file) for db_file in db_files]

    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    # prepare the results folder
    results_folder_addr = folder_addr / 'e2e' / 'pre_plots' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)

    time_bounds = []
    db_id = 0
    for result_database_file, time_mask in zip(result_database_files, time_masks):
        pacekt_analyzer = ULPacketAnalyzer(result_database_file)
        experiment_length_ts = pacekt_analyzer.last_ueip_ts - pacekt_analyzer.first_ueip_ts
        begin_ts = pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0]
        end_ts = pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1]
        time_bounds.append((begin_ts, end_ts))
        logger.info(f"Database {db_id}, experiment duration: {(experiment_length_ts)} seconds")
        logger.info(f"Database {db_id}, filtering packets from {begin_ts} to {end_ts}, length: {end_ts-begin_ts} seconds")
        db_id += 1

    # packet arrival dataset opening
    arrival_conf = e2e_config['packet_arrival']
    with open(folder_addr / 'packet_arrival' / 'datasets' / arrival_conf['dataset_name'] / 'config.json', 'r') as f:
        arrival_dataset_config = json.load(f)

    psize_eventtype_mapping = {int(k): int(v) for k, v in arrival_dataset_config['psize_eventtype_mapping'].items()}
    packet_arrival_events_arr = extract_packet_arrival_events(
        result_database_files, 
        time_bounds, 
        psize_eventtype_mapping, 
        arrival_dataset_config['filter_packet_sizes'],
        exp_config, 
        arrival_dataset_config['dtime_max']
    )

    # link quality dataset opening
    link_quality_conf = e2e_config['link_quality']
    with open(folder_addr / 'link_quality' / 'datasets' / link_quality_conf['dataset_name'] / 'config.json', 'r') as f:
        link_quality_dataset_config = json.load(f)

    link_retransmission_events_arr, link_mcs_events_arr = extract_link_quality_events(
        result_database_files, 
        time_bounds, 
        link_quality_dataset_config['stream_rntis'], 
        exp_config, 
        link_quality_dataset_config['dim_process_no_mcs'], 
        link_quality_dataset_config['min_mcs'], 
        link_quality_dataset_config['dtime_max']
    )

    # scheduling dataset opening
    scheduling_conf = e2e_config['scheduling']
    with open(folder_addr / 'scheduling' / 'datasets' / scheduling_conf['dataset_name'] / 'config.json', 'r') as f:
        scheduling_dataset_config = json.load(f)

    scheduling_events_arr = extract_scheduling_events(
        result_database_files, 
        time_bounds, 
        scheduling_dataset_config['stream_rntis'], 
        exp_config,
        scheduling_dataset_config['dtime_max']
    )

    prev_end_ts = 0
    arrivals_ts_list, arrivals_type_list = np.array([]), np.array([])
    link_ts_list, link_type_list = np.array([]), np.array([])
    depart_ts_list, scheduling_ts_list, scheduling_type_list = np.array([]), np.array([]), np.array([])
    for packet_arrival_events, link_retransmission_events, link_mcs_events, scheduling_events, time_bound in \
          zip(
              packet_arrival_events_arr, 
              link_retransmission_events_arr, 
              link_mcs_events_arr, 
              scheduling_events_arr, 
              time_bounds
              ):
        
        begin_ts, end_ts = time_bound





def plot_predictions(args):

    # read configuration from args.config
    dataset_config_path = Path(args.source) / "scheduling" / "datasets" / args.name / 'config.json'
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)
    
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
    p_num_rbs = []
    for batch in data['pred']:
        p_dtime.append(batch[0])
        p_num_rbs.append(batch[1])
    cp_prob = np.concatenate(p_dtime, axis=0)
    cp_num_rbs = np.concatenate(p_num_rbs, axis=0)

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
    cp_prob = np.exp(cp_prob[ar_index,:])
    # [1, 107]
    cp_num_rbs = np.exp(cp_num_rbs[ar_index,:])


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
        go.Scatter(x=ch_time[-2]+dtime_samples, y=cp_prob, mode='markers', name='predictions'),
        row=1, col=1,
        secondary_y=False
    )

    # add a bar plot, showing probabilities of the number of rbs
    cp_num_rbs = cp_num_rbs[0]
    fig.add_trace(
        go.Bar(x=np.arange(len(cp_num_rbs)), y=cp_num_rbs, name='Number of RBs', marker_color='rgba(0, 0, 255, 0.5)'),
        row=2, col=1
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