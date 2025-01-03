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


def link_quality_prediction(link_quality_runner, link_source_data,
                            packet_mcs_index, exp_config):

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    label_link_event = link_source_data[0][-1]
    prev_link_event = link_source_data[0][-2]

    logger.info(f"link_quality event prediction with sequence of size: {len(link_source_data[0])}")
    result = link_quality_runner.run(
        batch_size=1,
        source_data=link_source_data,
        return_predictions=True
    )
    p_dtime = []
    p_type = []
    for batch in result['pred']:
        p_dtime.append(batch[0])
        p_type.append(batch[1])
    cp_dtime_samples = np.concatenate(p_dtime, axis=0)
    cp_type_samples = np.concatenate(p_type, axis=0)
    pred_link_dtime = cp_dtime_samples[0,0]
    pred_link_type = cp_type_samples[0,0]
    pred_link_time_since_start = (prev_link_event['time_since_start'] + pred_link_dtime) % (max_num_frames*num_slots_per_frame*slots_duration_ms)

    # we have 4 rounds of retransmission: {0,1,2,3}
    # we have successful or unsuccessful RLC segment {0,1}
    # we use (total_hqrounds-1)+(rfailed*4) to map the event types to a unique number between 0 and 7
    # 'type_event' : int((item['total_hqrounds']-1)+(int(item['rfailed'])*4))
    pred_segment_mretx = pred_link_type % 4
    pred_segment_rfailed = int(np.floor(pred_link_type / 4))
    # consider this a retransmission event
    predicted_link_quality_event = {
        'idx_event' : -1, # we will fix it later
        'type_event' : int((pred_segment_mretx-1)+(int(pred_segment_rfailed)*4)),
        'time_since_start' : pred_link_time_since_start,
        'time_since_last_event' : pred_link_dtime,
        'mcs_index' : packet_mcs_index,
        'timestamp' : np.float64(prev_link_event['timestamp']) + np.float64(pred_link_dtime/1000.0)
    }

    logger.info(f"(last) pred link_dtime: {pred_link_dtime}, pred link_type_pred: {pred_link_type}, pred link_time_since_start: {pred_link_time_since_start}")
    logger.info(f"label link_dtime: {label_link_event['time_since_last_event']}, label link_type_pred: {label_link_event['type_event']}, label link_time_since_start: {label_link_event['time_since_start']}")
    #input()

    return predicted_link_quality_event, pred_segment_mretx, pred_segment_rfailed

def predict_scheduling(sched_runner, sched_history_sequence, exp_config) -> np.ndarray:
    """
    sched_history_sequence: [history_size, mcs_size, sched_sequence_length]
    For a certain arrival_history_sequence which is a batch of sequences of scheduling events,
    returns a list of predicted scheduling events (multiple samples) for the next segment_num segment.
    """

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    history_size = sched_history_sequence.shape[0]
    mcs_size = sched_history_sequence.shape[1]
    history_sequence_length = sched_history_sequence.shape[2]

    sched_history = np.empty((history_size, mcs_size, history_sequence_length+1), dtype=object)
    # append a dummy event to the end of each sequence to act as the label
    # [idx, idy] in [history_size]
    for idx in range(history_size):
        for idy in range(mcs_size):
            sched_history[idx, idy, :-1] = copy.deepcopy(sched_history_sequence[idx, idy, :])
            sched_history[idx, idy, -1] = {
                'idx_event' : 0, 'type_event': 0, 'slot' : 0, 'len' : 0, 'mcs_index' : 0, 
                'mretx' : 0, 'rfailed' : 0, 'num_rbs' : 0, 'num_symbols' : 0, 'time_since_start' : 0, 
                'time_since_last_event' : 0, 'timestamp' : 0
            }

    # combine sched_history first two dimensions
    # input is [history_size, mcs_size, history_sequence_length+1]
    # result is [history_size*mcs_size, history_sequence_length+1]
    sched_history = sched_history.reshape(-1, history_sequence_length+1)

    # predict the next segment
    result = sched_runner.run(
        batch_size=sched_history.shape[0],
        source_data=sched_history,
        data_specs={
            "num_event_types": 4, #FIXME
            "pad_token_id": 4, #FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )
    p_dtime = []
    p_len = []
    for batch in result['pred']:
        p_dtime.append(batch[0])
        p_len.append(batch[1])
    cp_dtime = np.concatenate(p_dtime, axis=1)
    cp_len = np.concatenate(p_len, axis=1)

    num_samples = cp_dtime.shape[0]
    cp_dtime = cp_dtime[:, :, 0] # [num_samples_dtime, history_size*mcs_size]
    cp_len = cp_len[:, :, 0] # [num_samples_len, history_size*mcs_size]
    last_history_events = sched_history_sequence[:, :, -1] # [history_size, mcs_size]

    cp_dtime = cp_dtime.reshape(num_samples, history_size, mcs_size) # [num_samples_len, history_size, mcs_size]
    cp_len = cp_len.reshape(num_samples, history_size, mcs_size) # [num_samples_len, history_size, mcs_size]
    assert cp_dtime.shape[1] == last_history_events.shape[0] # history_size should be the same
    assert cp_dtime.shape[2] == last_history_events.shape[1] # mcs_size should be the same
    assert cp_len.shape[0] == cp_dtime.shape[0] # sample numbers should be the same
    
    predictions = np.empty((num_samples, history_size, mcs_size), dtype=object)
    for idx in range(history_size):
        for idy in range(mcs_size):
            for idz in range(num_samples):
                pred_sched_dtime = cp_dtime[idz, idx, idy]
                pred_sched_len = int(cp_len[idz, idx, idy])
                prev_sched_event = last_history_events[idx, idy]

                # complete the prediction for this sample
                pred_segment_time_since_start = (prev_sched_event['time_since_start'] + pred_sched_dtime) % (max_num_frames*num_slots_per_frame*slots_duration_ms)
                pred_segment_slot = (prev_sched_event['slot'] + pred_sched_dtime/slots_duration_ms) % (num_slots_per_frame)
                pred_timestamp = np.float64(prev_sched_event['timestamp']) + np.float64(pred_sched_dtime/1000.0)

                predictions[idz, idx, idy] = {
                    'idx_event' : -1,
                    'type_event': -1,
                    'slot' : pred_segment_slot,
                    'len' : pred_sched_len,
                    'mcs_index' : -1,
                    'mretx' : -1,
                    'rfailed' : -1,
                    'num_rbs' : -1,
                    'num_symbols' : 3, # FIXME
                    'time_since_start' : pred_segment_time_since_start,
                    'time_since_last_event' : pred_sched_dtime,
                    'timestamp' : pred_timestamp
                }

    # input is [num_samples, history_size, mcs_size]
    # result is [num_samples*history_size, mcs_size]
    predictions = predictions.reshape(-1, mcs_size)
    return predictions


def predict_retx(retx_runner, retx_history_sequence, mcs_index : np.ndarray, num_rbs : np.ndarray, exp_config) -> np.ndarray:
    """
    input dims:
    retx_history_sequence dims: [retx_history_size, mcs_size, history_sequence_length]
    mcs_index dims: [mcs_size]
    num_rbs dims: [num_rbs_size, mcs_size]

    For a certain arrival_history_sequence which is a batch of sequences of scheduling events,
    returns a list of predicted scheduling events (multiple samples) for the next segment_num segment.
    """

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    history_size = retx_history_sequence.shape[0]
    history_sequence_length = retx_history_sequence.shape[2]
    mcs_size = mcs_index.shape[0]
    assert mcs_size == num_rbs.shape[1]
    assert mcs_size == retx_history_sequence.shape[1]
    num_rbs_size = num_rbs.shape[0]

    # input dimensions are [history_size, mcs_size, history_sequence_length]
    # output dimensions are [history_size, num_rbs_size, mcs_size, history_sequence_length+1]
    cond_retx_history = np.empty((history_size, num_rbs_size, mcs_size, history_sequence_length+1), dtype=object)
    # append a dummy event to the end of each sequence to act as the label
    # [idx, idy, idz] in [history_size, num_rbs_size, mcs_size]
    for idx in range(history_size):
        for idy in range(num_rbs_size):
            for idz in range(mcs_size):
                cond_retx_history[idx, idy, idz, :-1] = copy.deepcopy(retx_history_sequence[idx, idz, :])
                cond_retx_history[idx, idy, idz, -1] = {
                    'type_event': 0, #block attempt event
                    'timestamp': 0,
                    'time_since_start': 0,
                    'time_since_last_event': 0,
                    'rfailed': 0,
                    'mretx': 0,
                    'mcs_index': mcs_index[idz], # conditionally set
                    'num_rbs': num_rbs[idy, idz], # conditionally set
                }

    # combine cond_retx_history first two dimensions
    # input is [history_size, num_rbs_size, mcs_size, history_sequence_length+1]
    # result is [history_size*num_rbs_size*mcs_size, history_sequence_length+1]
    cond_retx_history = cond_retx_history.reshape(-1, history_sequence_length+1)

    # predict retx
    result = retx_runner.run(
        batch_size=cond_retx_history.shape[0],
        source_data=cond_retx_history,
        data_specs={
            "num_event_types": 2, #FIXME
            "pad_token_id": 2, #FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )
    p_retx = []
    p_rfailed = []
    for batch in result['pred']:
        p_retx.append(batch[0])
        p_rfailed.append(batch[1])
    cp_retx = np.concatenate(p_retx, axis=1)
    cp_rfailed = np.concatenate(p_rfailed, axis=1)
    cp_retx = cp_retx[:, :, 0] # [num_retx_samples, history_size*num_rbs_size*mcs_size]
    cp_rfailed = cp_rfailed[:, :, 0] # [num_retx_samples, history_size*num_rbs_size*mcs_size]
    num_retx_samples = cp_retx.shape[0]

    # input is [num_retx_samples, history_size*num_rbs_size*mcs_size]
    # result is [num_retx_samples, history_size, num_rbs_size, mcs_size]
    predictions_retx = cp_retx.reshape(num_retx_samples, history_size, num_rbs_size, mcs_size)
    predictions_rfailed = cp_rfailed.reshape(num_retx_samples, history_size, num_rbs_size, mcs_size)

    # input is [num_retx_samples, history_size, num_rbs_size, mcs_size]
    # result is [num_retx_samples*history_size, num_rbs_size, mcs_size]
    predictions_retx = cp_retx.reshape(-1, num_rbs_size, mcs_size)
    predictions_rfailed = cp_rfailed.reshape(-1, num_rbs_size, mcs_size)

    return predictions_retx, predictions_rfailed


def run_arrival_model(arrival_history_sequence, arrival_runner, exp_config) -> np.ndarray:
    """
    arrival_history_sequence: [batch_size, sequence_length]
    For a batch of arrival_history_sequence which is a batch of sequences of packet arrival events,
    returns predicted packet arrival events (multiple samples) for the next packet arrival.
    """
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    logger.info(f"packet arrival event prediction with sequence of shape: {arrival_history_sequence.shape}")
    result = arrival_runner.run(
        batch_size=arrival_history_sequence.shape[0],
        source_data=arrival_history_sequence,
        data_specs={
            "num_event_types": 10000, # a very large number
            "pad_token_id": 10000, # a very large number
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )
    p_dtime = []
    p_event_type = []
    for batch in result['pred']:
        p_dtime.append(batch[0])
        p_event_type.append(batch[1])
    cp_dtime = np.concatenate(p_dtime, axis=1)
    cp_event_type = np.concatenate(p_event_type, axis=1)

    cp_dtime = cp_dtime[:, :, 0] # [num_samples_dtime, batch_size]
    cp_event_type = cp_event_type[:, :, 0] # [num_samples_len, batch_size]
    last_history_events = arrival_history_sequence[:,-1] # [batch_size]
    batch_size = cp_dtime.shape[1]
    assert cp_dtime.shape[1] == last_history_events.shape[0] # batch sizes should be the same
    assert cp_event_type.shape[0] == cp_dtime.shape[0] # sample numbers should be the same
    num_samples = cp_dtime.shape[0]

    predictions = np.empty((num_samples, batch_size), dtype=object)
    for idx in range(batch_size):
        for idy in range(num_samples):
            # set the samples to proceed with
            pred_arrival_dtime = cp_dtime[idy, idx]
            pred_arrival_len = int(cp_event_type[idy, idx])
            prev_arrival_event = last_history_events[idx]

            pred_time_since_start = (prev_arrival_event['time_since_start'] + pred_arrival_dtime) % (max_num_frames*num_slots_per_frame*slots_duration_ms)
            pred_timestap = prev_arrival_event['timestamp'] + pred_arrival_dtime/1000
            predictions[idy,idx] = {
                'type_event' : int(pred_arrival_len),
                'time_since_start' : pred_time_since_start,
                'time_since_last_event' : pred_arrival_dtime,
                'timestamp' : pred_timestap
            }

    logger.info(f"predictions: {predictions}")
    return predictions

def predict_arrival(arrival_history_sequence, arrival_runner, num_packets, exp_config):
    
    # here the goal is to make a scheduling event from the predicted packet arrival event to be used in the scheduling prediction
    # that is why we need predicted_arrival_events and predicted_arrival_scheduling_events
    # in addition we keep the label_arrival_scheduling_events to compare the predictions
    # we create scheduling_dataset_segment0_ids to know the location of non-arrival scheduling events 
    # for each packet in dataset['label']['scheduling']
    predicted_arrival_events = []
    predicted_arrival_scheduling_events = []
    label_arrival_scheduling_events = [] 
    scheduling_dataset_segment0_ids = [ 0 ]
    for packet_num in range(num_packets):

        # make the arrival prediction
        # [num_samples_dtime, batch_size]
        arrival_predictions = run_arrival_model(arrival_history_sequence, arrival_runner, exp_config)
        predicted_arrival_scheduling_event_samples = []
        for predicted_arrival_event_sample in predicted_arrival_event_samples:
            # create the scheduling event from the predicted arrival event
            predicted_arrival_scheduling_event_sample = {
                'idx_event' : -1, # will be fixed later
                'type_event': 0, # must be zero
                'slot' : -1, # will be fixed later
                'len' : int(predicted_arrival_event_sample['type_event']),
                'mcs_index' : label_arrival_scheduling_event['mcs_index'], # FIXME! cheating maybe?
                'mretx' : 0, # don't need
                'rfailed' : 0, # don't need
                'num_rbs' : 0, # don't need
                'num_symbols' : 0, # don't need
                'time_since_start' : predicted_arrival_event_sample['time_since_start'],
                'time_since_last_event' : -1, # will be fixed later
                'timestamp' : predicted_arrival_event_sample['timestamp']
            }
            predicted_arrival_scheduling_event_samples.append(predicted_arrival_scheduling_event_sample)
    
        # for now we only use the first sample FIXME!
        predicted_arrival_events.append(predicted_arrival_event_samples[0])
        predicted_arrival_scheduling_events.append(predicted_arrival_scheduling_event_samples[0])

        logger.info(f"Predicted packet {packet_num} scheduling arrival event: {predicted_arrival_scheduling_events[-1]}")
        logger.info(f"Label packet {packet_num} scheduling arrival event: {label_arrival_scheduling_events[-1]}")
        #input()

        # remove the label event, append the predicted arrival event for the next prediction
        arrival_source_data[0] = arrival_source_data[0][:-1]
        arrival_source_data[0].append(predicted_arrival_events[-1])

    return predicted_arrival_events, predicted_arrival_scheduling_events, label_arrival_scheduling_events, scheduling_dataset_segment0_ids

def calc_num_rbs(mcs_index : np.ndarray, len_bytes : np.ndarray):
    """
    mcs_index: [ mcs_size ]
    len_bytes: [ len_size, mcs_size ]
    returns num_rbs: [ len_size, mcs_size ]
    """
    mcs_size = mcs_index.shape[0]
    len_size = len_bytes.shape[0]
    num_rbs = np.zeros((len_size, mcs_size), dtype=int)
    for idx in range(len_size):
        for idy in range(mcs_size):
            if len_bytes[idx, idy] <= 20:
                num_rbs[idx, idy] = 5
            elif len_bytes[idx, idy] > 20:
                num_rbs[idx, idy] = 26
    return num_rbs


def e2e_delay_prediction(
        data,
        arrival_runner, 
        mcs_runner, 
        retx_runner,
        sched_runner, 
        exp_config,
        num_packets,
        filter_successful_attempts = True,
        segment_dimension_limit = 100,
        exclude_link_quality = True,
        max_num_segments = 5
    ):

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']

    # arrival_history_sequence = np.array([ data['arrival'] ])
    # predict all packet arrivals as it is not dependent on link quality or scheduling
    #predicted_arrival_events, predicted_arrival_scheduling_events, \
    #    label_arrival_scheduling_events, scheduling_dataset_segment0_ids = predict_arrivals(
    #        arrival_history_sequence, arrival_runner, num_packets, exp_config
    #    )

    # now we start predicting the scheduling and link quality events
    # after every scheduling prediction, check if predicted packet arrivals should be included in the source_data

    # create the waiting_packets_queue, first packet arrival event is from history
    #waiting_packets_queue = [ data['scheduling'][-1] ]
    #packet_0_scheduling_event = waiting_packets_queue[0]

    sched_history_sequence = np.array([
        [ data['scheduling' ] ]
    ]) # [sched_history_size, mcs_size, sched_sequence_length]

    mcs_history_sequence = np.array([
        [ data['mcs' ] ] 
    ]) # [mcs_history_size, mcs_size, mcs_sequence_length]

    retx_history_sequence = np.array([
        [ data['retx' ] ]
    ] ) # [retx_history_size, mcs_size, retx_sequence_length]

    packet0_mcs_index = data['scheduling'][-1]['mcs_index']
    mcs_index = np.array(
        [ packet0_mcs_index ]
    ) # [ mcs_size ]

    # start scheduling prediciton for the segments
    predicted_packet_transmissions = []
    predicted_segment_events = []
    for segment_num in range(max_num_segments):
        packet_num = len(predicted_packet_transmissions)
        
        # figure all sizes
        mcs_size = mcs_index.shape[0]
        mcs_history_size = mcs_history_sequence.shape[0]
        mcs_sequence_length = mcs_history_sequence.shape[2]
        retx_history_size = retx_history_sequence.shape[0]
        retx_sequence_length = retx_history_sequence.shape[2]
        sched_history_size = sched_history_sequence.shape[0]
        sched_sequence_length = sched_history_sequence.shape[2]

        logger.info(f"Predicting packet {packet_num}, segment: {segment_num+1}, history sequence dims - mcs: {mcs_history_sequence.shape}, retx: {retx_history_sequence.shape}, sched: {sched_history_sequence.shape}")

        # predict the scheduling event
        # the result is the segment's scheduling event, but mretx, rfailed, mcs, and segment number are not set
        # we either use link quality prediction for that or set them to 0
        # the prediction gives a list of samples
        # input: sched_history_sequence with dims [sched_history_size, mcs_size, sched_sequence_length]
        # output: [sched_pred_size = num_sched_samples*sched_history_size, mcs_size]
        sched_predictions = predict_scheduling(
            sched_runner, 
            sched_history_sequence, 
            exp_config
        )
        sched_pred_size = sched_predictions.shape[0]
        len_bytes = np.empty((sched_pred_size, mcs_size), dtype=object)
        for idx in range(sched_pred_size):
            for idy in range(mcs_size):
                len_bytes[idx,idy] = sched_predictions[idx,idy]['len']
                sched_predictions[idx,idy]['type_event'] = segment_num + 1
                sched_predictions[idx,idy]['mcs_index'] = mcs_index[idy]
                if exclude_link_quality:
                    sched_predictions[idx,idy]['mretx'] = 0
                    sched_predictions[idx,idy]['rfailed'] = 0

        logger.info(f"sched predictions shape: {sched_predictions.shape}")

        # calc num_rbs for the segment predictions
        # len_bytes dims: [sched_pred_size, mcs_size]
        # num_rbs_size: [sched_pred_size, mcs_size]
        num_rbs = calc_num_rbs(mcs_index, len_bytes)

        logger.info(f"num_rbs shape: {num_rbs.shape}")

        # predict the retx events
        if not exclude_link_quality:
            # predict link quality event
            # input dims:
            # retx_history_sequence dims: [retx_history_size, mcs_size, sched_sequence_length]
            # mcs_index dims: [mcs_size]
            # num_rbs dims: [sched_pred_size, mcs_size]
            predictions_retx, predictions_rfailed = predict_retx(
                retx_runner, retx_history_sequence,  mcs_index, num_rbs, exp_config
            )
            # output dims: [retx_pred_size = num_retx_samples*retx_history_size, sched_pred_size, mcs_size]
            retx_pred_size = predictions_retx.shape[0]

            logger.info(f"retx predictions shape: {predictions_retx.shape}")

            # now set the mretx and rfailed for the segment predictions
            segment_predictions = np.empty((retx_pred_size, sched_pred_size, mcs_size), dtype=object)
            for idx in range(retx_pred_size):
                for idy in range(sched_pred_size):
                    for idz in range(mcs_size):
                        segment_predictions[idx,idy,idz] = copy.deepcopy(sched_predictions[idy,idz])
                        segment_predictions[idx,idy,idz]['mretx'] = predictions_retx[idx,idy,idz]
                        segment_predictions[idx,idy,idz]['rfailed'] = predictions_retx[idx,idy,idz]
                        segment_predictions[idx,idy,idz]['num_rbs'] = num_rbs[idy,idz]

            # size becomes [ segment_pred_size = retx_pred_size*sched_pred_size, mcs_size]
            segment_predictions = segment_predictions.reshape(-1, mcs_size)
        else:
            for idx in range(sched_pred_size):
                for idy in range(mcs_size):
                    sched_predictions[idx,idy]['num_rbs'] = num_rbs[idx,idy]
            # size becomes [ segment_pred_size = sched_pred_size, mcs_size]
            segment_predictions = sched_predictions

    
        # segment predictions have dim: [segment_pred_size, mcs_size]
        logger.info(f"segment predictions shape: {segment_predictions.shape}")

        # dimension reduction
        # randomly select segment_dimension_limit samples out of segment_pred_size
        # if segment_pred_size is less than segment_dimension_limit, then use all
        segment_pred_size = segment_predictions.shape[0]
        if segment_pred_size > segment_dimension_limit:
            segment_pred_size = segment_dimension_limit
            selected_indices = np.random.choice(segment_predictions.shape[0], segment_dimension_limit, replace=False)
            segment_predictions = segment_predictions[selected_indices]
        
        # segment predictions have dim: [segment_pred_size, mcs_size]
        logger.info(f"reduced dim sched predictions shape: {segment_predictions.shape}")

        # append the segment predictions to the scheduling history sequences
        upd_sched_history = np.empty((sched_history_size, segment_pred_size, mcs_size, sched_sequence_length), dtype=object)
        for idx in range(sched_history_size):
            for idy in range(segment_pred_size):
                for idz in range(mcs_size):
                    upd_sched_history[idx,idy,idz,:] = np.append(
                        copy.deepcopy(sched_history_sequence[idx, idz, 1:]), # remove the oldest event
                        copy.deepcopy(segment_predictions[idy,idz]) # append the new event
                    )
                    for pos in range(sched_sequence_length):
                        upd_sched_history[idx,idy,idz,pos]['idx_event'] = pos
        upd_sched_history = upd_sched_history.reshape(-1, mcs_size, sched_sequence_length)

        logger.info(f"updated sched history shape: {upd_sched_history.shape}")

        # append the segment predictions to retx history sequences
        upd_retx_history = np.empty((retx_history_size, segment_pred_size, mcs_size, retx_sequence_length), dtype=object)
        for idx in range(retx_history_size):
            for idy in range(segment_pred_size):
                for idz in range(mcs_size):
                    block_event = copy.deepcopy(segment_predictions[idy,idz])
                    block_event['type_event'] = 0
                    block_event['time_since_last_event'] = (block_event['timestamp'] - retx_history_sequence[idx, idz, -1]['timestamp'])*1000
                    del block_event['len'] 
                    del block_event['slot']
                    del block_event['num_symbols']
                    upd_retx_history[idx,idy,idz,:] = np.append(
                        copy.deepcopy(retx_history_sequence[idx, idz, 1:]), # remove the oldest event
                        block_event # append the new event
                    )
                    for pos in range(retx_sequence_length):
                        upd_retx_history[idx,idy,idz,pos]['idx_event'] = pos
        upd_retx_history = upd_retx_history.reshape(-1, mcs_size, retx_sequence_length)

        logger.info(f"updated retx history shape: {upd_retx_history.shape}")

        # append the segment predictions to mcs history sequences
        # mcs_history_sequence dim: [mcs_history_size, mcs_size, mcs_sequence_length]
        upd_mcs_history = np.empty((segment_pred_size, mcs_history_size, mcs_size, mcs_sequence_length), dtype=object)
        for idx in range(segment_pred_size):
            for idy in range(mcs_history_size):
                for idz in range(mcs_size):
                    block_event = copy.deepcopy(segment_predictions[idx,idz])
                    block_event['type_event'] = 0
                    block_event['time_since_last_event'] = (block_event['timestamp'] - mcs_history_sequence[idy, idz, -1]['timestamp'])*1000
                    del block_event['len'] 
                    del block_event['slot']
                    del block_event['num_symbols']
                    upd_mcs_history[idx,idy,idz,:] = np.append(
                        copy.deepcopy(mcs_history_sequence[idy, idz, 1:]), # remove the oldest event
                        block_event # append the new event
                    )
                    for pos in range(mcs_sequence_length):
                        upd_mcs_history[idx,idy,idz,pos]['idx_event'] = pos
        upd_mcs_history = upd_mcs_history.reshape(-1, mcs_size, mcs_sequence_length)

        logger.info(f"updated mcs history shape: {upd_mcs_history.shape}")
        input()         

        # update the history sequences
        mcs_history_sequence = upd_mcs_history
        retx_history_sequence = upd_retx_history
        sched_history_sequence = upd_sched_history


    return predicted_packet_transmissions

"""
"s63_e2e" : {
    "gpu" : -1,
    "batch_size" : 32,
    "dataset_name" : "test1",
    "arrival" : {
        "trained_model_name" : "test0",
        "trained_model_id" : "1105474_140407072232064_241202-120400",
        "probability_generation": {
            "num_steps_dtime": 1000,
            "sample_dtime_max" : 40.3,
            "sample_dtime_min" : 39.7,
            "num_steps_event_type": 100,
            "sample_event_type_max" : 90,
            "sample_event_type_min" : 50
    
        },
        "num_samples_dtime": 10,
        "num_samples_event_type": 10
    },
    "scheduling" : {
        "trained_model_name" : "test0",
        "trained_model_id" : "1112063_140079729623680_241202-162330",
        "probability_generation": {
            "num_steps_dtime": 1000,
            "sample_dtime_max" : 15,
            "sample_dtime_min" : 0,
            "num_steps_len": 500,
            "sample_len_max" : 300,
            "sample_len_min" : 0
    
        },
        "num_samples_dtime": 100,
        "num_samples_len": 100     
    },
    "mcs" : {
        "trained_model_name" : "test0",
        "trained_model_id" : "1106489_139985643180672_241202-123840",
        "num_samples_mcs": 1000
    },
    "retx" : {
        "trained_model_name" : "test0",
        "trained_model_id" : "1106489_139985643180672_241202-123840",
        "num_samples_mretx": 1000,
        "num_samples_rfailed": 1000
    }
}
"""


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
        predicted_packet_transmissions = e2e_delay_prediction(
            data = entry,
            arrival_runner = arrival_runner, 
            mcs_runner = mcs_runner, 
            retx_runner = retx_runner,
            sched_runner = sched_runner, 
            exp_config = exp_config,
            filter_successful_attempts = True,
            segment_dimension_limit = 10,
            num_packets = 5,
            exclude_link_quality = False,
            max_num_segments = 5
        )
        print(predicted_packet_transmissions)
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