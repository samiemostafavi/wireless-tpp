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
                "thinning": arrival_conf['thinning'] if 'thinning' in arrival_conf else {},
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            },
            "prediction_config" : arrival_conf
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id)
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
    config = Config.build_from_dict(config, experiment_id=experiment_id)
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
                "thinning": scheduling_conf['thinning'] if 'thinning' in scheduling_conf else {},
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            },
            "prediction_config" : scheduling_conf
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id)
    return config

def create_packet_arrival_window(packet_arrival_events, arrival_event, history_window_size, num_post_packet_arrival_events):
    # find the index of the arrival_event in the packet_arrival_events
    idx = packet_arrival_events.index(arrival_event)+1
    if idx < history_window_size:
        return None, None
    events_window = []
    idx_pre = idx
    # we don't include the label event in the window, that comes in the post arrival events
    for event in packet_arrival_events[idx_pre-history_window_size:idx_pre]:
        events_window.append(
            {
                'idx_event' : -1, # will be fixed later
                'type_event': event['type_event'],
                'time_since_start' : event['time_since_start'],
                'time_since_last_event' : event['time_since_last_event'],
                'timestamp' : event['timestamp']
            }
        )

    if len(events_window) != history_window_size:
        return None, None

    post_packet_arrival_events = []
    idx_post = idx
    while idx_post < len(packet_arrival_events) and len(post_packet_arrival_events) < num_post_packet_arrival_events:
        event = packet_arrival_events[idx_post]
        post_packet_arrival_events.append(
            {
                'idx_event' : -1, # will be fixed later
                'type_event': event['type_event'],
                'time_since_start' : event['time_since_start'],
                'time_since_last_event' : event['time_since_last_event'],
                'timestamp' : event['timestamp']
            }
        )
        idx_post += 1

    if len(post_packet_arrival_events) != num_post_packet_arrival_events:
        return None, None

    return events_window, post_packet_arrival_events

def create_link_quality_window(sorted_link_events, arrival_event, dim_process_no_mcs, history_window_size, num_post_link_quality_events):

    # find the first non-mcs event right after the arrival_event['timestamp']
    idx = 0
    for idx, event in enumerate(sorted_link_events):
        #  event['type_event'] < dim_process_no_mcs makes sure the label event is not mcs event
        if event['timestamp'] >= arrival_event['timestamp'] and event['type_event'] < dim_process_no_mcs:
            break

    if idx < history_window_size:
        return None, None
    
    events_window = []
    idx_pre = idx
    # we don't include the label event in the window, that comes in the post link events
    for event in sorted_link_events[idx_pre-history_window_size:idx_pre]:
        events_window.append(
            {
                'idx_event' : -1, # will be fixed later
                'type_event': event['type_event'],
                'time_since_start' : event['time_since_start'],
                'time_since_last_event' : event['time_since_last_event'],
                'timestamp' : event['timestamp'],
                'mcs_index' : event['mcs_index'] if event['type_event'] < dim_process_no_mcs else None
            }
        )

    if len(events_window) != history_window_size:
        return None, None
    
    post_link_events = []
    # keep the current arrival event in the post scheduling events as it holds departure time
    idx_post = idx
    while idx_post < len(sorted_link_events) and len(post_link_events) < num_post_link_quality_events:
        event = sorted_link_events[idx_post]
        post_link_events.append(
            {
                'idx_event' : -1, # will be fixed later
                'type_event': event['type_event'],
                'time_since_start' : event['time_since_start'],
                'time_since_last_event' : event['time_since_last_event'],
                'timestamp' : event['timestamp'],
                'mcs_index' : event['mcs_index'] if event['type_event'] < dim_process_no_mcs else None
            }
        )
        idx_post += 1

    if len(post_link_events) != num_post_link_quality_events:
        return None, None

    return events_window, post_link_events

def create_scheduling_window(sorted_scheduling_events, arrival_event, history_window_size, num_post_scheduling_events):

    # find the arrival event in the sorted_scheduling_events, from timestamp
    idx = 0
    found = False
    for idx, event in enumerate(sorted_scheduling_events):
        if event['timestamp'] == arrival_event['timestamp']:
            found = True
            break
    if not found:
        logger.warning("Arrival event not found in the segment_events")
        return None, None

    if idx < history_window_size or idx+1 >= len(sorted_scheduling_events):
        return None, None

    events_window = []
    # we don't include the label event in the window, that comes in the post link events
    idx_pre = idx
    while idx_pre > 0 and len(events_window) < history_window_size:
        event = sorted_scheduling_events[idx_pre]
        events_window.append(
            {
                'idx_event' : -1, # will be fixed later
                'type_event': event['segment']+1,
                'slot' : event['slot'],
                'len' : event['len'],
                'mcs_index' : event['mcs_index'],
                'mac_retx' : event['mac_retx'],
                'rlc_failed' : event['rlc_failed'],
                'num_rbs' : event['num_rbs'],
                'num_symbols' : event['num_symbols'],
                'time_since_start' : event['time_since_start'],
                'time_since_last_event' : event['time_since_last_event'],
                'timestamp' : event['timestamp'],
            }
        )
        idx_pre -= 1

    if len(events_window) != history_window_size:
        return None, None
    
    # sort the events_window based on timestamp as it is reveresed
    events_window = sorted(events_window, key=lambda x: x['timestamp'], reverse=False)

    post_scheduling_events = []
    idx_post = idx + 1
    while idx_post < len(sorted_scheduling_events) and len(post_scheduling_events) < num_post_scheduling_events:
        event = sorted_scheduling_events[idx_post]
        post_scheduling_events.append(
            {
                'idx_event' : -1, # will be fixed later
                'type_event': event['segment']+1,
                'slot' : event['slot'],
                'len' : event['len'],
                'mcs_index' : event['mcs_index'],
                'mac_retx' : event['mac_retx'],
                'rlc_failed' : event['rlc_failed'],
                'num_rbs' : event['num_rbs'],
                'num_symbols' : event['num_symbols'],
                'time_since_start' : event['time_since_start'],
                'time_since_last_event' : event['time_since_last_event'],
                'timestamp' : event['timestamp'],
            }
        )
        idx_post += 1

    if len(post_scheduling_events) != num_post_scheduling_events:
        return None, None

    return events_window, post_scheduling_events


def scheduling_prediction(scheduling_runner, scheduling_analyzer, scheduling_source_data,
                          segment_num, packet_mcs_index, num_symbols, 
                          exp_config) -> list:
    """
    For a certain scheduling_source_data which is a batch of sequences of scheduling events,
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

    label_scheduling_event = scheduling_source_data[0][-1]
    prev_scheduling_event = scheduling_source_data[0][-2]

    logger.info(f"scheduling event prediction with sequence of size: {len(scheduling_source_data[0])}")
    # predict the next segment
    result = scheduling_runner.run(
        batch_size=1,
        source_data=scheduling_source_data,
        return_predictions=True
    )
    p_dtime = []
    p_num_rbs = []
    for batch in result['pred']:
        p_dtime.append(batch[0])
        p_num_rbs.append(batch[1])
    cp_dtime_samples = np.concatenate(p_dtime, axis=1)
    cp_num_rbs_samples = np.concatenate(p_num_rbs, axis=1)
    
    batch_num = 0
    pred_segment_dtime_samples = cp_dtime_samples[:, batch_num, 0]
    pred_segment_num_rbs_samples = cp_num_rbs_samples[:, batch_num, 0]
    logger.info(f"scheduling prediction num samples: {len(pred_segment_dtime_samples)}")
    predicted_segment_event_samples = []
    for pred_segment_dtime, pred_segment_num_rbs in zip(pred_segment_dtime_samples, pred_segment_num_rbs_samples):
        # set the samples to proceed with
        pred_segment_dtime = pred_segment_dtime
        pred_segment_num_rbs = int(pred_segment_num_rbs)

        # complete the prediction for this sample
        pred_segment_time_since_start = (prev_scheduling_event['time_since_start'] + pred_segment_dtime) % (max_num_frames*num_slots_per_frame*slots_duration_ms)
        pred_segment_slot = (prev_scheduling_event['slot'] + pred_segment_dtime/slots_duration_ms) % (num_slots_per_frame)
        pred_timestamp = np.float64(prev_scheduling_event['timestamp']) + np.float64(pred_segment_dtime/1000.0)

        # get the segment length in bytes from mcs, num_rbs, and num_symbols
        pred_segment_tbs = scheduling_analyzer.figure_mcs_to_tbs(packet_mcs_index, pred_segment_num_rbs, num_symbols, SCHED_OFFSET_S=scheduling_time_ahead_ms/1000.0)
        if pred_segment_tbs < 0:
            logger.error(f"packet_mcs_index: {packet_mcs_index}, pred_segment_num_rbs: {pred_segment_num_rbs}, num_symbols: {num_symbols}, pred_segment_tbs: {pred_segment_tbs}")
            return None
        pred_segment_len_bytes = pred_segment_tbs - 32 # 32 is the overhead FIXME!

        predicted_segment_event = {
            'idx_event' : -1,
            'type_event': segment_num + 1,
            'slot' : pred_segment_slot,
            'len' : pred_segment_len_bytes,
            'mcs_index' : packet_mcs_index,
            'mac_retx' : -1,
            'rlc_failed' : -1,
            'num_rbs' : pred_segment_num_rbs,
            'num_symbols' : num_symbols,
            'time_since_start' : pred_segment_time_since_start,
            'time_since_last_event' : pred_segment_dtime,
            'timestamp' : pred_timestamp
        }

        predicted_segment_event_samples.append(predicted_segment_event)

    logger.info(f"(last) pred segment_dtime: {pred_segment_dtime}, pred segment_num_rbs: {pred_segment_num_rbs}, pred segment_time_since_start: {pred_segment_time_since_start}, pred segment_slot: {pred_segment_slot}, pred_timestamp: {pred_timestamp}")
    logger.info(f"label segment_dtime: {label_scheduling_event['time_since_last_event']}, label segment_num_rbs: {label_scheduling_event['num_rbs']}, label segment_time_since_start: {label_scheduling_event['time_since_start']}, label segment_slot: {label_scheduling_event['slot']}, label_timestamp: {label_scheduling_event['timestamp']}")
        #input()

    logger.info(f"(last) pred mcs_index: {packet_mcs_index}, pred num_symbols:{num_symbols}, pred segment_len_bytes: {pred_segment_len_bytes}")
    logger.info(f"label mcs_index: {label_scheduling_event['mcs_index']}, num_symbols:{label_scheduling_event['num_symbols']}, label segment_len_bytes: {label_scheduling_event['len']}")
    #input()

    return predicted_segment_event_samples


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
    # we use (total_hqrounds-1)+(rlc_failed*4) to map the event types to a unique number between 0 and 7
    # 'type_event' : int((item['total_hqrounds']-1)+(int(item['rlc_failed'])*4))
    pred_segment_mac_retx = pred_link_type % 4
    pred_segment_rlc_failed = int(np.floor(pred_link_type / 4))
    # consider this a retransmission event
    predicted_link_quality_event = {
        'idx_event' : -1, # we will fix it later
        'type_event' : int((pred_segment_mac_retx-1)+(int(pred_segment_rlc_failed)*4)),
        'time_since_start' : pred_link_time_since_start,
        'time_since_last_event' : pred_link_dtime,
        'mcs_index' : packet_mcs_index,
        'timestamp' : np.float64(prev_link_event['timestamp']) + np.float64(pred_link_dtime/1000.0)
    }

    logger.info(f"(last) pred link_dtime: {pred_link_dtime}, pred link_type_pred: {pred_link_type}, pred link_time_since_start: {pred_link_time_since_start}")
    logger.info(f"label link_dtime: {label_link_event['time_since_last_event']}, label link_type_pred: {label_link_event['type_event']}, label link_time_since_start: {label_link_event['time_since_start']}")
    #input()

    return predicted_link_quality_event, pred_segment_mac_retx, pred_segment_rlc_failed

def packet_arrival_event_prediction(arrival_source_data, arrival_model_runner, exp_config) -> list:
    """
    For a certain arrival_source_data which is a batch of sequences of packet arrival events,
    returns a list of predicted packet arrival events (multiple samples) for the next packet arrival.
    """
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    label_arrival_event = arrival_source_data[0][-1]
    prev_arrival_event = arrival_source_data[0][-2]

    logger.info(f"packet arrival event prediction with sequence of size: {len(arrival_source_data[0])}")
    result = arrival_model_runner.run(
        batch_size=1,
        source_data=arrival_source_data,
        return_predictions=True
    )
    p_dtime = []
    p_event_type = []
    for batch in result['pred']:
        p_dtime.append(batch[0])
        p_event_type.append(batch[1])
    cp_dtime = np.concatenate(p_dtime, axis=1)
    cp_event_type = np.concatenate(p_event_type, axis=1)

    batch_num = 0
    pred_arrival_dtime_samples = cp_dtime[:, batch_num, 0]
    pred_arrival_type_samples = cp_event_type[:, batch_num, 0]

    logger.info(f"arrival prediction num samples: {len(pred_arrival_dtime_samples)}")
    predicted_packet_arrival_event_samples = []
    for pred_arrival_dtime, pred_arrival_type in zip(pred_arrival_dtime_samples, pred_arrival_type_samples):
        # set the samples to proceed with
        pred_arrival_dtime = pred_arrival_dtime
        pred_arrival_type = int(pred_arrival_type)

        pred_time_since_start = (prev_arrival_event['time_since_start'] + pred_arrival_dtime) % (max_num_frames*num_slots_per_frame*slots_duration_ms)
        pred_timestap = prev_arrival_event['timestamp'] + pred_arrival_dtime/1000
        predicted_packet_arrival_event = {
            'type_event' : int(pred_arrival_type),
            'time_since_start' : pred_time_since_start,
            'time_since_last_event' : pred_arrival_dtime,
            'timestamp' : pred_timestap
        }
        predicted_packet_arrival_event_samples.append(predicted_packet_arrival_event)

    logger.info(f"pred arrival_dtime: {pred_arrival_dtime}, pred arrival_type: {pred_arrival_type}, pred arrival_time_since_start: {pred_time_since_start}")
    logger.info(f"label arrival_dtime: {label_arrival_event['time_since_last_event']}, label arrival_type: {label_arrival_event['type_event']}, label arrival_time_since_start: {label_arrival_event['time_since_start']}")
    #input()

    return predicted_packet_arrival_event_samples

def packet_arrival_prediction(dataset, arrival_model_runner, num_packets, exp_config, eventtype_psize_mapping):
    
    # here the goal is to make a scheduling event from the predicted packet arrival event to be used in the scheduling prediction
    # that is why we need predicted_arrival_events and predicted_arrival_scheduling_events
    # in addition we keep the label_arrival_scheduling_events to compare the predictions
    # we create scheduling_dataset_segment0_ids to know the location of non-arrival scheduling events 
    # for each packet in dataset['label']['scheduling']
    arrival_source_data = [ dataset['history']['packet_arrival'] ]
    predicted_arrival_events = []
    predicted_arrival_scheduling_events = []
    label_arrival_scheduling_events = [] 
    scheduling_dataset_segment0_ids = [ 0 ]
    for packet_num in range(num_packets):

        # find the label scheduling event for packet_num-th packet arrival
        counter, label_arrival_scheduling_event = 0, None
        for idx, event in enumerate(dataset['label']['scheduling']):
            if event['type_event'] == 0:
                if counter == packet_num:
                    label_arrival_scheduling_event = event
                    scheduling_dataset_segment0_ids.append(idx+1)
                    break
                counter += 1
        if label_arrival_scheduling_event is None:
            logger.error(f"No label arrival scheduling event found for packet {packet_num}")
            return None
        label_arrival_scheduling_events.append(label_arrival_scheduling_event)

        # find the label arrival event for packet_num-th packet
        label_arrival_event = dataset['label']['packet_arrival'][packet_num]

        # append the label event to the source_data for prediction
        arrival_source_data[0].append(label_arrival_event)
        arrival_source_data[0] = arrival_source_data[0][1:]
        for pos, event in enumerate(arrival_source_data[0]):
            event['idx_event'] = pos

        # make the arrival prediction
        predicted_arrival_event_samples = packet_arrival_event_prediction(arrival_source_data, arrival_model_runner, exp_config)
        predicted_arrival_scheduling_event_samples = []
        for predicted_arrival_event_sample in predicted_arrival_event_samples:
            # create the scheduling event from the predicted arrival event
            predicted_arrival_scheduling_event_sample = {
                'idx_event' : -1, # will be fixed later
                'type_event': 0, # must be zero
                'slot' : -1, # will be fixed later
                'len' : eventtype_psize_mapping[int(predicted_arrival_event_sample['type_event'])],
                'mcs_index' : label_arrival_scheduling_event['mcs_index'], # FIXME! cheating maybe?
                'mac_retx' : 0, # don't need
                'rlc_failed' : 0, # don't need
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

def e2e_delay_prediction(dataset, 
                         arrival_model_runner, link_quality_runner, scheduling_runner, 
                         exp_config, dim_process_no_mcs, scheduling_analyzer, eventtype_psize_mapping,
                         dont_predict_link_events=True, link_event_close_enough_ms=2, max_num_segments=5, num_packets=1):

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']

    # maybe cheating? FIXME! we assume the same number of symbols (3) for all segments
    num_symbols = dataset['label']['scheduling'][0]['num_symbols']

    # predict all packet arrivals as it is not dependent on link quality or scheduling
    predicted_arrival_events, predicted_arrival_scheduling_events, \
        label_arrival_scheduling_events, scheduling_dataset_segment0_ids = packet_arrival_prediction(
            dataset, arrival_model_runner, num_packets, exp_config, eventtype_psize_mapping
        )

    # now we start predicting the scheduling and link quality events
    # after every scheduling prediction, check if predicted packet arrivals should be included in the source_data

    # create the waiting_packets_queue, first packet arrival event is from history
    waiting_packets_queue = [ dataset['history']['scheduling'][-1] ]
    packet_0_scheduling_event = waiting_packets_queue[0]

    scheduling_source_data = [dataset['history']['scheduling']]
    link_source_data = [dataset['history']['link_quality']]
    prev_predicted_link_event = {}

    # start scheduling prediciton for the segments
    sum_departed_bytes = 0
    predicted_packet_transmissions = []
    predicted_segment_events = []
    last_label_link_event_idx = -1
    segment_num = -1
    while True:
        segment_num += 1

        packet_num = len(predicted_packet_transmissions)
        logger.info(f"Predicting packet {packet_num} with size: {packet_0_scheduling_event['len']}, segment: {segment_num+1}, sum_departed_bytes so far: {sum_departed_bytes}")

        # fix the source_data for scheduling
        # append the label event to the source_data
        # remove the first event from the source_data
        label_scheduling_event = dataset['label']['scheduling'][scheduling_dataset_segment0_ids[packet_num] + segment_num]
        scheduling_source_data[0].append(label_scheduling_event)
        scheduling_source_data[0] = scheduling_source_data[0][1:]
        for pos, event in enumerate(scheduling_source_data[0]):
            event['idx_event'] = pos

        # fix the source_data for the link quality
        # we move one step ahead in the link quality events, if a previous prediction was actually used
        # also when a new packet is being evaluated for the first segment, we move one step ahead (FIXME! doesn't seem right)
        if (not dont_predict_link_events) and \
            ( (prev_predicted_link_event != {}) or (segment_num == 0) ):
            logger.info(f"Setting a new label link event")
            label_link_event = None
            for idx, event in enumerate(dataset[0]['label']['link_quality']):
                if idx > last_label_link_event_idx and event['type_event'] < dim_process_no_mcs:
                    label_link_event = event
                    last_label_link_event_idx = idx
                    break
            if label_link_event is None:
                logger.error("No label link event found")
                return None
            link_source_data[0].append(label_link_event)
            link_source_data[0] = link_source_data[0][1:]
            for pos, event in enumerate(link_source_data[0]):
                event['idx_event'] = pos

        # predict the scheduling event
        # the result is the segment's scheduling event, but mac_retx and rlc_failed are not set
        # we either use link quality prediction for that or set them to 0
        # the prediction gives a list of samples
        predicted_segment_event_samples = scheduling_prediction(
                scheduling_runner, scheduling_analyzer, scheduling_source_data, 
                segment_num, packet_0_scheduling_event['mcs_index'], num_symbols, exp_config
            )
        # for now we use the first sample
        predicted_segment_event = predicted_segment_event_samples[0]

        predicted_segment_event['mac_retx'] = 0
        predicted_segment_event['rlc_failed'] = 0
        prev_predicted_link_event = {}
        if not dont_predict_link_events:
            # predict link quality event
            predicted_link_quality_event, pred_segment_mac_retx, pred_segment_rlc_failed = link_quality_prediction(
                link_quality_runner, link_source_data, 
                packet_0_scheduling_event['mcs_index'], exp_config
            )
            # if link_time_since_start is close enough to time_since_start e.g. 2ms
            if abs(predicted_link_quality_event['timestamp'] - predicted_segment_event['timestamp']) < (link_event_close_enough_ms/1000.0):
                # remove the label event, append the prediction event
                link_source_data[0] = link_source_data[0][:-1]
                link_source_data[0].append(predicted_link_quality_event)
            
                # now complete the predicted segment event and append
                predicted_segment_event['mac_retx'] = pred_segment_mac_retx
                predicted_segment_event['rlc_failed'] = pred_segment_rlc_failed
                prev_predicted_link_event = predicted_link_quality_event
            
            logger.info(f"Predicted link event: {prev_predicted_link_event}")

        # segment prediction completes here
        predicted_segment_events.append(predicted_segment_event)

        # fix the scheduling source_data for the next segment
        # remove the segment label event, append the prediction event
        scheduling_source_data[0] = scheduling_source_data[0][:-1]
        scheduling_source_data[0].append(predicted_segment_event)

        # update the sum_departed_bytes
        sum_departed_bytes += ( predicted_segment_event['len'] if not predicted_segment_event['rlc_failed'] else 0 )

        # check if a packet is completed
        if sum_departed_bytes >= packet_0_scheduling_event['len']:
            logger.info(f"Packet {len(predicted_packet_transmissions)} transmission completed")
            #input()
            # we have completed packet_0's transmission
            predicted_packet_transmissions.append({
                'packet_scheduling_event' : packet_0_scheduling_event,
                'predicted_segments' : predicted_segment_events,
            })
            if len(predicted_packet_transmissions) >= num_packets:
                break
            
            # if during the process of completing packet_0, we have received more packets
            # it means we have interleaved packets
            # in such case num_departed_bytes is subtracted by packet_0_scheduling_event['len']
            # and the rest of it will be used for the next packet(s)
            if len(waiting_packets_queue) > 1:
                # interleaved packets case
                logger.info(f"Interleaved packets, sum_departed_bytes: {sum_departed_bytes}, completed packet size: {packet_0_scheduling_event['len']}, remaining: {sum_departed_bytes - packet_0_scheduling_event['len']}, next packet size: {waiting_packets_queue[1]['len']}")
                #input()
                # make a deep copy of predicted_segment_event
                copyof_predicted_segment_event = copy.deepcopy(predicted_segment_event)
                sum_departed_bytes -= packet_0_scheduling_event['len']
                if sum_departed_bytes < 0:
                    logger.error(f"sum_departed_bytes: {sum_departed_bytes}")
                    return None
                copyof_predicted_segment_event['segment_num'] = 1
                predicted_segment_events = [ copyof_predicted_segment_event ]
                segment_num = 0

                # remove the head in the queue and set it as the new packet_0_scheduling_event
                waiting_packets_queue = waiting_packets_queue[1:]
                packet_0_scheduling_event = waiting_packets_queue[0]
            else:
                # no interleaved packets
                # reset the counters and proceed with the next packet
                sum_departed_bytes = 0
                predicted_segment_events = [] 
                segment_num = -1

                # get the next packet arrival event (predicted) and set it as the new packet_0_scheduling_event
                copyof_next_predicted_arrival_event = copy.deepcopy(
                    predicted_arrival_scheduling_events[len(predicted_packet_transmissions)-1]
                )
                # here we have to fix some attributes in the packet_0_scheduling_event
                copyof_next_predicted_arrival_event['time_since_last_event'] = (
                        copyof_next_predicted_arrival_event['timestamp'] - predicted_segment_event['timestamp']
                    ) * 1000
                copyof_next_predicted_arrival_event['slot'] = int( 
                        predicted_segment_event['slot'] + (copyof_next_predicted_arrival_event['time_since_last_event']/slots_duration_ms)
                    ) % num_slots_per_frame
                waiting_packets_queue = [ copyof_next_predicted_arrival_event ]
                packet_0_scheduling_event = waiting_packets_queue[0]

                # append the arrival event to the scheduling source_data
                scheduling_source_data[0].append(packet_0_scheduling_event)
                scheduling_source_data[0] = scheduling_source_data[0][1:]
                for pos, event in enumerate(scheduling_source_data[0]):
                    event['idx_event'] = pos
                
                # report the new packet arrival event
                logger.info(f"New pred packet arrival scheduling event: {packet_0_scheduling_event}")
                logger.info(f"Its label packet arrival scheduling event: {label_arrival_scheduling_events[len(predicted_packet_transmissions)-1]}")
                #input()           

    return predicted_packet_transmissions

def generate_predictions(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        e2e_config = json.load(f)
    
    time_masks = e2e_config['time_masks']
    num_post_scheduling_events = e2e_config['num_post_scheduling_events']
    num_post_packet_arrival_events = e2e_config['num_post_packet_arrival_events']
    num_post_link_quality_events = e2e_config['num_post_link_quality_events']

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
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    # prepare the results folder
    results_folder_addr = folder_addr / 'e2e' / 'prediction_results' / args.name
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

    # for mcs_to_bytes function
    scheduling_analyzer = ULSchedulingAnalyzer(
        total_prbs_num = total_prbs_num, 
        symbols_per_slot = symbols_per_slot,
        slots_per_frame = num_slots_per_frame, 
        slots_duration_ms = slots_duration_ms, 
        scheduling_map_num_integers = scheduling_map_num_integers,
        max_num_frames = max_num_frames,
        db_addr = result_database_files[0]
    )

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

    arrival_window_size = arrival_dataset_config['window_config']['size']
    link_window_size = link_quality_dataset_config['window_config']['size']
    dim_process_no_mcs = link_quality_dataset_config['dim_process_no_mcs']
    scheduling_window_size = scheduling_dataset_config['window_config']['size']

    dataset = []
    for packet_arrival_events, link_retransmission_events, link_mcs_events, scheduling_events, time_bound in \
          zip(
              packet_arrival_events_arr, 
              link_retransmission_events_arr, 
              link_mcs_events_arr, 
              scheduling_events_arr, 
              time_bounds
              ):
        
        begin_ts, end_ts = time_bound

        # sort events
        packet_arrival_events = sorted(packet_arrival_events, key=lambda x: x['timestamp'], reverse=False)
        link_quality_events = sorted([ *link_retransmission_events, *link_mcs_events ], key=lambda x: x['timestamp'], reverse=False)
        scheduling_events = sorted(scheduling_events, key=lambda x: x['timestamp'], reverse=False)

        # we sweep over the packet arrival events, then we create joint history windows when there is sufficient data
        for arrival_event in packet_arrival_events:
            packet_arrival_history, post_packet_arrival_events = create_packet_arrival_window(
                packet_arrival_events, arrival_event, 
                arrival_window_size, num_post_packet_arrival_events
            )
            link_quality_history, post_link_quality_events = create_link_quality_window(
                link_quality_events, arrival_event, 
                dim_process_no_mcs, link_window_size, num_post_link_quality_events
            )
            scheduling_history, post_scheduling_events = create_scheduling_window(
                scheduling_events, arrival_event, 
                scheduling_window_size, num_post_scheduling_events
            )
            if packet_arrival_history and link_quality_history and scheduling_history:
                # we can make a prediction
                dataset.append({
                    'label': {
                        'packet_arrival' : post_packet_arrival_events,
                        'link_quality' : post_link_quality_events,
                        'scheduling' : post_scheduling_events,
                    },
                    'history': {
                        'packet_arrival' : packet_arrival_history,
                        'link_quality' : link_quality_history,
                        'scheduling' : scheduling_history
                    }
                })

    # figure a few common parameters
    batch_size = e2e_config['batch_size']
    gpu = e2e_config['gpu']
    e2e_config['method'] = args.predict

    # load arrival model configuration
    arrival_exp_config = create_arrival_exp_config(args, arrival_conf, batch_size, gpu, results_folder_addr)
    arrival_model_runner = TPPRunnerPacketArrival(
        runner_config=arrival_exp_config,
        unique_model_dir=False
    )
    # load link quality model configuration
    link_quality_exp_config = create_link_quality_exp_config(args, link_quality_conf, batch_size, gpu, results_folder_addr)
    link_quality_runner = TPPRunnerLinkQuality(
        runner_config=link_quality_exp_config,
        unique_model_dir=False
    )
    # load scheduling model configuration
    scheduling_exp_config = create_scheduling_exp_config(args, scheduling_conf, batch_size, gpu, results_folder_addr)
    scheduling_runner = TPPRunnerScheduling(
        runner_config=scheduling_exp_config,
        unique_model_dir=False
    )

    # swap keys and values in psize_eventtype_mapping
    eventtype_psize_mapping = {v: k for k, v in psize_eventtype_mapping.items()}

    # run e2e delay prediction
    # for each entry in dataset, we run one prediction
    for entry in dataset:
        predicted_packet_transmissions = e2e_delay_prediction(
            dataset = entry,
            arrival_model_runner = arrival_model_runner, 
            link_quality_runner = link_quality_runner, 
            scheduling_runner = scheduling_runner, 
            exp_config = exp_config,
            dim_process_no_mcs = dim_process_no_mcs, 
            scheduling_analyzer = scheduling_analyzer,
            dont_predict_link_events = True,
            eventtype_psize_mapping = eventtype_psize_mapping,
            link_event_close_enough_ms = 2, 
            max_num_segments = 5, 
            num_packets = 5
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
    # label_dtime, label_time, label_type, slot_seqs, len_seqs, mcs_seqs, mac_retx_seqs, rlc_failed_seqs, num_rbs_seqs
    # data['label'] dimensions: [num batches, 8 attributes , batch size, seq length]

    h_dtime, h_time, h_event_type, h_slot, h_len, h_mcs, h_mac_retx, h_rlc_failed, h_num_rbs = [],[],[],[],[],[],[],[],[]
    history_mcs_data = []
    for batch in data['label']:
        h_dtime.append(batch[0])
        h_time.append(batch[1])
        h_event_type.append(batch[2])
        h_slot.append(batch[3])
        h_len.append(batch[4])
        h_mcs.append(batch[5])
        h_mac_retx.append(batch[6])
        h_rlc_failed.append(batch[7])
        h_num_rbs.append(batch[8])

    ch_dtime = np.concatenate(h_dtime, axis=0)
    ch_time = np.concatenate(h_time, axis=0)
    ch_event_type = np.concatenate(h_event_type, axis=0)
    ch_slot = np.concatenate(h_slot, axis=0)
    ch_len = np.concatenate(h_len, axis=0)
    ch_mcs = np.concatenate(h_mcs, axis=0)
    ch_mac_retx = np.concatenate(h_mac_retx, axis=0)
    ch_rlc_failed = np.concatenate(h_rlc_failed, axis=0)
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
    ch_mac_retx = ch_mac_retx[ar_index,:]
    ch_rlc_failed = ch_rlc_failed[ar_index,:]
    ch_num_rbs = ch_num_rbs[ar_index,:]

    logger.info(f"Event types in the history plus the label: {ch_event_type}")

    # [num probability samples]
    cp_prob = np.exp(cp_prob[ar_index,:])
    # [1, 107]
    cp_num_rbs = np.exp(cp_num_rbs[ar_index,:])


    # history packets time series
    packet_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mrtx_list = np.array([ch_mac_retx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_rrtx_list = np.array([ch_rlc_failed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])

    # history segments time series
    segment_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_type_list = np.array([ch_event_type[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_mrtx_list = np.array([ch_mac_retx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_rrtx_list = np.array([ch_rlc_failed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
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
    # label_dtime, label_time, label_type, slot_seqs, len_seqs, mcs_seqs, mac_retx_seqs, rlc_failed_seqs, num_rbs_seqs
    # data['label'] dimensions: [num batches, 8 attributes , batch size, seq length]

    h_dtime, h_time, h_event_type, h_slot, h_len, h_mcs, h_mac_retx, h_rlc_failed, h_num_rbs = [],[],[],[],[],[],[],[],[]
    history_mcs_data = []
    for batch in data['label']:
        h_dtime.append(batch[0])
        h_time.append(batch[1])
        h_event_type.append(batch[2])
        h_slot.append(batch[3])
        h_len.append(batch[4])
        h_mcs.append(batch[5])
        h_mac_retx.append(batch[6])
        h_rlc_failed.append(batch[7])
        h_num_rbs.append(batch[8])

    ch_dtime = np.concatenate(h_dtime, axis=0)
    ch_time = np.concatenate(h_time, axis=0)
    ch_event_type = np.concatenate(h_event_type, axis=0)
    ch_slot = np.concatenate(h_slot, axis=0)
    ch_len = np.concatenate(h_len, axis=0)
    ch_mcs = np.concatenate(h_mcs, axis=0)
    ch_mac_retx = np.concatenate(h_mac_retx, axis=0)
    ch_rlc_failed = np.concatenate(h_rlc_failed, axis=0)
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
    ch_mac_retx = ch_mac_retx[ar_index,:]
    ch_rlc_failed = ch_rlc_failed[ar_index,:]
    ch_num_rbs = ch_num_rbs[ar_index,:]

    logger.info(f"Event types in the history plus the label: {ch_event_type}")

    # cp_dtime_samples and cp_num_rbs_samples have the shape: [num gen samples, num samples in all the batches, 1]
    cp_dtime = np.mean(cp_dtime_samples[:, ar_index, 0])
    # [1, 107]
    cp_num_rbs = np.mean(cp_num_rbs_samples[:, ar_index, 0])


    # history packets time series
    packet_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mrtx_list = np.array([ch_mac_retx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_rrtx_list = np.array([ch_rlc_failed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_mcs_list = np.array([ch_mcs[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])
    packet_ts_list = np.array([ch_time[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] == 0])

    # history segments time series
    segment_len_list = np.array([ch_len[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_type_list = np.array([ch_event_type[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_mrtx_list = np.array([ch_mac_retx[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
    segment_rrtx_list = np.array([ch_rlc_failed[idx] for idx, _ in enumerate(ch_dtime) if ch_event_type[idx] > 0])
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