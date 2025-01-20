import pickle
import random, copy
import os, sys, json
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from edaf.core.uplink.analyze_packet import ULPacketAnalyzer
from edaf.core.uplink.analyze_channel import ULChannelAnalyzer
from edaf.core.uplink.analyze_scheduling import ULSchedulingAnalyzer

from src.link_quality import extract_link_quality_events, window_history_segment_events, window_history_mcs_decision_events
from src.packet_arrival import extract_packet_arrival_events, window_history_arrival_events
from src.scheduling import extract_scheduling_events, window_history_scheduling_events

if not os.getenv('DEBUG'):
    logger.remove()
    logger.add(sys.stdout, level="INFO")


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

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']

    db_id = 0
    prev_end_ts = 0
    arrivals_ts_list, arrivals_type_list = np.array([]), np.array([])
    link_segment_type_list, link_segment_ts_list, link_mcs_type_list, link_mcs_ts_list = np.array([]), np.array([]), np.array([]), np.array([])
    depart_ts_list, scheduling_ts_list, scheduling_type_list = np.array([]), np.array([]), np.array([])
    for result_database_file, time_mask in zip(result_database_files, time_masks):
        packet_analyzer = ULPacketAnalyzer(result_database_file)
        scheduling_analyzer = ULSchedulingAnalyzer(
            total_prbs_num = total_prbs_num, 
            symbols_per_slot = symbols_per_slot,
            slots_per_frame = num_slots_per_frame, 
            slots_duration_ms = slots_duration_ms, 
            scheduling_map_num_integers = scheduling_map_num_integers,
            max_num_frames = max_num_frames,
            db_addr = result_database_file
        )
        chan_analyzer = ULChannelAnalyzer(result_database_file)

        # figure the time bounds
        experiment_length_ts = packet_analyzer.last_ueip_ts - packet_analyzer.first_ueip_ts
        begin_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0]
        end_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1]
        logger.info(f"Database {db_id}, experiment duration: {(experiment_length_ts)} seconds")
        logger.info(f"Database {db_id}, filtering packets from {begin_ts} to {end_ts}, length: {end_ts-begin_ts} seconds")
        
        # figure the stream rnti
        packets = packet_analyzer.figure_packettx_from_ts(begin_ts, begin_ts+1.0) # just take one second of packets
        packets_rnti_set = set([item['rlc.attempts'][0]['rnti'] for item in packets])
        packets_rnti_set.discard(None)
        if len(packets_rnti_set) > 1:
            logger.error("Multiple RNTIs in the packet stream, exiting...")
            return
        stream_rnti = list(packets_rnti_set)[0]

        # extract the events
        # packet arrival events
        packet_arrival_events = extract_packet_arrival_events(
            packet_analyzer, scheduling_analyzer, begin_ts, end_ts, exp_config, dtime_max = np.inf
        )
        # link quality events
        link_segment_events, link_mcs_events = extract_link_quality_events(
            chan_analyzer, packet_analyzer, 
            scheduling_analyzer, stream_rnti,
            begin_ts, end_ts, exp_config,
            mcs_event_type = 'change', 
            mcs_eval_interval_ms = 100
        )
        # scheduling events
        scheduling_events = extract_scheduling_events(
            packet_analyzer, scheduling_analyzer, begin_ts, end_ts, exp_config
        )

        # create time series for plotting
        arrivals_type_list = np.concatenate((arrivals_type_list, np.array([item['type_event'] for item in packet_arrival_events])))
        arrivals_ts_list = np.concatenate((arrivals_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in packet_arrival_events])))

        # segment events
        link_segment_type_list = np.concatenate((link_segment_type_list, np.array([item['mretx']+4*item['rfailed'] for item in link_segment_events])))
        link_segment_ts_list = np.concatenate((link_segment_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in link_segment_events])))

        # mcs events
        link_mcs_type_list = np.concatenate((link_mcs_type_list, np.array([item['mcs_index'] for item in link_mcs_events])))
        link_mcs_ts_list = np.concatenate((link_mcs_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in link_mcs_events])))

        # scheduling events
        scheduling_type_list = np.concatenate((scheduling_type_list, np.array([item['segment']+1 for item in scheduling_events])))
        scheduling_ts_list = np.concatenate((scheduling_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in scheduling_events])))
        depart_ts_list = np.concatenate((depart_ts_list, np.array([(item['depart_timestamp']-begin_ts+prev_end_ts)*1000 for item in scheduling_events if item['depart_timestamp']>0])))

        db_id += 1
        prev_end_ts = (end_ts-begin_ts) + prev_end_ts


    fig = make_subplots(rows=3, cols=1, subplot_titles=("Packet Arrival", "Link Quality", "Scheduling"))
    # Add the scatter plot for predictions to the first row
    fig.add_trace(
        go.Scatter(
            x=arrivals_ts_list, 
            y=np.ones(len(arrivals_ts_list)),
            mode='markers+text', 
            text=arrivals_type_list,
            textposition='top center',
            name='History arrival events'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=link_segment_ts_list, 
            y=np.ones(len(link_segment_ts_list)),
            mode='markers+text', 
            text=link_segment_type_list,
            textposition='top center',
            name='History segment events'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=link_mcs_ts_list, 
            y=np.ones(len(link_mcs_ts_list)),
            mode='markers+text', 
            text=link_mcs_type_list,
            textposition='top center',
            name='History mcs events'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=scheduling_ts_list, 
            y=np.ones(len(scheduling_ts_list)),
            mode='markers+text', 
            text=scheduling_type_list,
            textposition='top center',
            name='History scheduling events'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=depart_ts_list, 
            y=1.5*np.ones(len(depart_ts_list)),
            mode='markers+text', 
            text=np.zeros(len(depart_ts_list)),
            textposition='top center',
            name='History departure events'
        ),
        row=3, col=1
    )

    fig.update_layout(
        title='Plot',
        xaxis_title='Time [ms]',
        yaxis_title='Values',
        legend_title='Legend',
    )
    fig.update_xaxes(title_text='Time [ms]', row=1, col=1)
    fig.update_yaxes(title_text='Values', row=1, col=1)
    fig.update_xaxes(title_text='Time [ms]', row=2, col=1)
    fig.update_yaxes(title_text='Values', row=2, col=1)
    fig.update_xaxes(title_text='Time [ms]', row=3, col=1)
    fig.update_yaxes(title_text='Values', row=3, col=1)
    fig.update_xaxes(matches='x')
    fig.write_html(str(results_folder_addr / 'combined_plot.html'))


def find_event(input_event, list_of_events, condition = None):
    closest_event_index, closest_event = min(
        enumerate(list_of_events), 
        key=lambda x: abs(x[1]['timestamp'] - input_event['timestamp'])
    )
    # checks
    if abs(closest_event['timestamp'] - input_event['timestamp']) > 1e-4:
        logger.error(f"Could not find a close event for the input event: {input_event}, closest event: {closest_event}")
        return None, None
    if condition:
        if closest_event[condition['key']] != condition['value']:
            logger.error("The event founded does not meet the condition.")
            return None, None
    return closest_event, closest_event_index


def find_last_event_before_input(input_event, list_of_events, condition = None):
    if condition is None:
        last_event_index, last_event = max(
            enumerate(list_of_events), 
            key=lambda x: x[1]['timestamp'] if x[1]['timestamp'] <= input_event['timestamp'] else -float('inf')
        )
    else:
        last_event_index, last_event = max(
            enumerate(list_of_events), 
            key=lambda x: x[1]['timestamp'] if x[1]['timestamp'] <= input_event['timestamp'] and x[1][condition['key']] == condition['value'] else -float('inf')
        )
    # checks
    if last_event['timestamp'] > input_event['timestamp']:
        logger.error("Could not find any event before the input event.")
        return None, None
    return last_event, last_event_index

def find_next_event_after_input(input_event, list_of_events, condition = None):
    if condition is None:
        next_event_index, next_event = min(
            enumerate(list_of_events), 
            key=lambda x: x[1]['timestamp'] if x[1]['timestamp'] > input_event['timestamp'] else float('inf')
        )
    else:
        next_event_index, next_event = min(
            enumerate(list_of_events), 
            key=lambda x: x[1]['timestamp'] if x[1]['timestamp'] > input_event['timestamp'] and x[1][condition['key']] == condition['value'] else float('inf')
        )
    # checks
    if next_event['timestamp'] < input_event['timestamp']:
        logger.error("Could not find any event before the input event.")
        return None, None
    return next_event, next_event_index


def create_training_dataset(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    # select the source configuration
    e2e_config = dataset_config[args.configname]
    time_masks = e2e_config['time_masks']

    # read experiment configuration
    folder_addr = Path(args.source)
    # find all .db files in the folder
    db_files = list(folder_addr.glob("*.db"))
    if not db_files:
        logger.error("No database files found in the specified folder.")
        return
    result_database_files = [str(db_file) for db_file in db_files]
    logger.info(f"Found {len(result_database_files)} database files in the folder: {result_database_files}")

    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    # read dataset configurations
    num_future_packets = e2e_config['num_future_packets']
    arrival_dataset_config = e2e_config['arrival']
    scheduling_dataset_config = e2e_config['scheduling']
    mcs_dataset_config = e2e_config['mcs']
    retx_dataset_config = e2e_config['retx']

    # prepare the results folder
    results_folder_addr = folder_addr / 'e2e' / 'datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)

    # prepare experiment configurations
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']

    dataset = []
    db_id = 0
    for time_mask_entry in time_masks:
        result_database_file = folder_addr / str(time_mask_entry[0])
        if str(result_database_file) not in result_database_files:
            logger.warning(f"Database {result_database_file} not found in the folder, skipping...")
            continue
        time_mask = time_mask_entry[1]
        logger.info(f"Processing database {result_database_file}")
        packet_analyzer = ULPacketAnalyzer(result_database_file)
        scheduling_analyzer = ULSchedulingAnalyzer(
            total_prbs_num = total_prbs_num, 
            symbols_per_slot = symbols_per_slot,
            slots_per_frame = num_slots_per_frame, 
            slots_duration_ms = slots_duration_ms, 
            scheduling_map_num_integers = scheduling_map_num_integers,
            max_num_frames = max_num_frames,
            db_addr = result_database_file
        )
        chan_analyzer = ULChannelAnalyzer(result_database_file)

        # figure the time bounds
        experiment_length_ts = packet_analyzer.last_ueip_ts - packet_analyzer.first_ueip_ts
        begin_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0]
        end_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1]
        logger.info(f"Database {db_id}, experiment duration: {(experiment_length_ts)} seconds")
        logger.info(f"Database {db_id}, filtering packets from {begin_ts} to {end_ts}, length: {end_ts-begin_ts} seconds")
        
        # figure the stream rnti
        packets = packet_analyzer.figure_packettx_from_ts(begin_ts, begin_ts+1.0) # just take one second of packets
        packets_rnti_set = set([item['rlc.attempts'][0]['rnti'] for item in packets])
        packets_rnti_set.discard(None)
        if len(packets_rnti_set) > 1:
            logger.error("Multiple RNTIs in the packet stream, exiting...")
            return
        stream_rnti = list(packets_rnti_set)[0]

        # extract the events for these time bounds
        # packet arrival events
        packet_arrival_events = extract_packet_arrival_events(
            packet_analyzer, scheduling_analyzer, begin_ts, end_ts, exp_config, dtime_max = np.inf
        )
        # link quality events
        link_segment_events, link_mcs_events = extract_link_quality_events(
            chan_analyzer, packet_analyzer, scheduling_analyzer, stream_rnti,
            begin_ts, end_ts, exp_config, 
            mcs_event_type = mcs_dataset_config['mcs_event_type'], 
            mcs_eval_interval_ms = mcs_dataset_config['mcs_eval_interval_ms']
        )
        # scheduling events
        scheduling_events = extract_scheduling_events(
            packet_analyzer, scheduling_analyzer, begin_ts, end_ts, exp_config
        )

        if mcs_dataset_config['filter_successful_attempts']:
            filtered_segment_events = [item for item in link_segment_events if (item['mretx'] > 0 or item['rfailed'])]
        filtered_link_events = [ *filtered_segment_events, *link_mcs_events ]
        filtered_link_events = sorted(filtered_link_events, key=lambda x: x['timestamp'], reverse=False)

        dataset_this_db = []
        # now we iterate backwards over arrival events in packet_arrival_events
        L = num_future_packets
        for n in range(len(packet_arrival_events) - 1, -1, -1):
            # go back at least L+1 events: +1 so we are sure we have recorded everything
            if n > len(packet_arrival_events) - L - 2:
                continue

            print(f"\rProcessing packet {n + 1}/{len(packet_arrival_events)} ({100.0 - ((n + 1) / len(packet_arrival_events) * 100):.2f}%)", end="")

            arrival_event_n = packet_arrival_events[n]
            arrival_event_nL = packet_arrival_events[n+L]
            arrival_history_sequence = window_history_arrival_events(n+1, packet_arrival_events, arrival_dataset_config['window_config']['history'])
            if len(arrival_history_sequence) == 0:
                continue

            # find arrival_event in scheduling_events
            sched_event_m, m = find_event(arrival_event_n, scheduling_events, {'key': 'segment', 'value': -1})
            sched_event_mL, mL = find_event(arrival_event_nL, scheduling_events, {'key': 'segment', 'value': -1})
            if sched_event_m is None or sched_event_mL is None:
                continue

            # first segment of packet n
            sched_event_m1 = scheduling_events[m+1]
            sched_history_sequence = window_history_scheduling_events(m+1, scheduling_events, scheduling_dataset_config['window_config']['history'])
            if len(sched_history_sequence) == 0:
                continue

            # find it in link_segment_events
            segment_event_l1, l1 = find_event(sched_event_m1, link_segment_events)
            if segment_event_l1 is None:
                continue
            retx_history_sequence = window_history_segment_events(l1, link_segment_events, retx_dataset_config['window_config']['history'])
            if len(retx_history_sequence) == 0:
                continue

            # find the label mcs event
            link_event_mcs_k1, mcs_k1 = find_next_event_after_input(sched_event_m, filtered_link_events, {'key': 'type_event', 'value': 1})
            if link_event_mcs_k1 is None:
                continue
            mcs_history_sequence = window_history_mcs_decision_events(mcs_k1, filtered_link_events, mcs_dataset_config['window_config']['history'])
            if len(mcs_history_sequence) == 0:
                continue

            # label sequence comes from sched_events
            label_sequence = [item for item in scheduling_events[m+1:mL+1]]
            dataset_this_db.append({
                'n' : n,
                'arrival' : arrival_history_sequence[:-1], # we don't want the label event
                'scheduling' : sched_history_sequence[:-1],
                'mcs' : mcs_history_sequence[:-1],
                'retx' : retx_history_sequence[:-1],
                'label' : label_sequence
            })        
        print("\n", end="")

        # print length of dataset
        logger.info(f"Number of total entries produced by db {db_id} dataset: {len(dataset_this_db)}")
        if len(dataset_this_db) > 0:
            print(dataset_this_db[0])

        # append elements of one_db_dataset to dataset
        dataset.extend(dataset_this_db)
        db_id += 1
    
    # shuffle the dataset
    random.shuffle(dataset)

    logger.success(f"Number of total entries in the dataset: {len(dataset)}")

    # prepare the results folder
    results_folder_addr = folder_addr / 'e2e' / 'datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(e2e_config, indent=4)
        f.write(json_obj)

    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'test.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    
def window_fulltf_dataset_events(db_dataset, only_arrivals, history_len, prediction_len):
    """
    Create a training dataset
    """
    # find arrival events indices in the dataset
    arrival_indices = [idx for idx, event in enumerate(db_dataset) if event['segment'] == -1]
    training_db_dataset = []
    if only_arrivals:
        max_num_segments = 0
        for mp in range(len(arrival_indices) - 2, -1, -1):
            if mp > len(arrival_indices) - prediction_len - 1:
                continue
            if mp <= history_len-1:
                break
            sequence = []
            
            for mpp in range(history_len+prediction_len):
                mdx = mp-history_len+mpp
                idx = arrival_indices[mdx]
                idxp1 = arrival_indices[mdx+1]
                db_dataset[idx]['label_mask'] = 0 if mpp < history_len else 1
                db_dataset[idx]['interarrival_time'] = \
                    ( db_dataset[idxp1]['timestamp'] - db_dataset[idx]['timestamp'] ) * 1000
                sequence.append( copy.deepcopy(db_dataset[idx]) )
            training_db_dataset.append(sequence)
    else:
        max_num_segments = max([event['segment']+1 for event in db_dataset])
        for mp in range(len(arrival_indices) - 2, -1, -1):
            if mp > len(arrival_indices) - prediction_len:
                continue
            m = arrival_indices[mp]
            if m <= history_len-1:
                break
            sequence = []
            # append segment events
            for mpp in range(history_len):
                idx = m - history_len + mpp
                sequence.append( copy.deepcopy(db_dataset[idx]) )
            # append arrivals
            for mpp in range(prediction_len):
                idx = arrival_indices[mp+mpp]
                sequence.append( copy.deepcopy(db_dataset[idx]) )
            training_db_dataset.append(sequence)
        
    num_arrivals = len(arrival_indices)
    return training_db_dataset, num_arrivals, max_num_segments


def window_fulltf_dataset_time(db_dataset, only_arrivals, history_len, prediction_len):
    """
    Create a training dataset
    """
    # find arrival events indices in the dataset
    arrival_indices = [idx for idx, event in enumerate(db_dataset) if event['segment'] == -1]
    last_arrival_ts = db_dataset[arrival_indices[-1]]['timestamp']
    first_arrival_ts = db_dataset[arrival_indices[0]]['timestamp']
    first_event_ts = db_dataset[0]['timestamp']
    training_db_dataset = []
    if only_arrivals:
        max_num_segments = 0
        for mp in range(len(arrival_indices) - 2, -1, -1):
            mp_ts = db_dataset[arrival_indices[mp]]['timestamp']
            if last_arrival_ts - mp_ts < prediction_len:
                continue
            if mp_ts - first_event_ts < history_len:
                break
            db_dataset[arrival_indices[mp]]['interarrival_time'] = \
                    ( db_dataset[arrival_indices[mp+1]]['timestamp'] - db_dataset[arrival_indices[mp]]['timestamp'] ) * 1000
            sequence = [ copy.deepcopy(db_dataset[arrival_indices[mp]]) ] # append the arrival event itself
            sequence[-1]['label_mask'] = 1 # prediction
            # history (only arrivals)
            mp2 = mp
            tmp_ts = db_dataset[arrival_indices[mp2-1]]['timestamp']
            while mp_ts - tmp_ts <= history_len:
                mp2 -= 1
                if mp2 == 0:
                    break
                db_dataset[arrival_indices[mp2]]['interarrival_time'] = \
                    ( db_dataset[arrival_indices[mp2+1]]['timestamp'] - db_dataset[arrival_indices[mp2]]['timestamp'] ) * 1000
                sequence.append( copy.deepcopy(db_dataset[arrival_indices[mp2]]) )
                sequence[-1]['label_mask'] = 0 # history
                tmp_ts = db_dataset[arrival_indices[mp2-1]]['timestamp']
            # prediction (only arrivals)
            mp2 = mp
            tmp_ts = db_dataset[arrival_indices[mp2+1]]['timestamp']
            while tmp_ts - mp_ts <= prediction_len:
                mp2 += 1
                if mp2 >= len(arrival_indices)-1:
                    break
                db_dataset[arrival_indices[mp2]]['interarrival_time'] = \
                    ( db_dataset[arrival_indices[mp2+1]]['timestamp'] - db_dataset[arrival_indices[mp2]]['timestamp'] ) * 1000
                sequence.append( copy.deepcopy(db_dataset[arrival_indices[mp2]]) )
                sequence[-1]['label_mask'] = 1 # prediction
                tmp_ts = db_dataset[arrival_indices[mp2+1]]['timestamp']
            # sort sequence based on 'timestamp'
            sequence = sorted(sequence, key=lambda x: x['timestamp'])
            training_db_dataset.append(sequence)
    else:
        max_num_segments = max([event['segment']+1 for event in db_dataset])
        for mp in range(len(arrival_indices) - 2, -1, -1):
            mp_ts = db_dataset[arrival_indices[mp]]['timestamp']
            if last_arrival_ts - mp_ts < prediction_len:
                continue
            if mp_ts - first_arrival_ts < history_len:
                break
            sequence = [ copy.deepcopy(db_dataset[arrival_indices[mp]]) ] # append the arrival event itself
            # history (all events)
            np = arrival_indices[mp]
            tmp_ts = db_dataset[np-1]['timestamp']
            while mp_ts - tmp_ts <= history_len:
                np -= 1
                if np == 0:
                    break
                sequence.append( copy.deepcopy(db_dataset[np]) )
                tmp_ts = db_dataset[np-1]['timestamp']
            # prediction (only arrivals)
            mp2 = mp
            tmp_ts = db_dataset[arrival_indices[mp2+1]]['timestamp']
            while tmp_ts - mp_ts <= prediction_len:
                mp2 += 1
                if mp2 >= len(arrival_indices)-1:
                    break
                sequence.append( copy.deepcopy(db_dataset[arrival_indices[mp2]]) )
                tmp_ts = db_dataset[arrival_indices[mp2+1]]['timestamp']
            # sort sequence based on 'timestamp'
            sequence = sorted(sequence, key=lambda x: x['timestamp'])
            training_db_dataset.append(sequence)

    num_arrivals = len(arrival_indices)
    return training_db_dataset, num_arrivals, max_num_segments


def create_fulltf_training_subdataset(args):
    """
    Create a training dataset
    """

    # read configuration from args.config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    # select the source configuration
    dataset_config = dataset_config[args.configname]
    main_ds_name = dataset_config["main_ds_name"]
    
    # read experiment configuration
    folder_addr = Path(args.source)

    # this means we have a main dataset and now we need to create training datasets
    dataset_size = dataset_config["dataset_size_max"]
    split_ratios = dataset_config["split_ratios"]

    # open the dataset in the same folder with name 'main_ds_name'
    dataset_pickle_file = folder_addr / 'e2e' / 'datasets' / main_ds_name / 'dataset.pkl'
    with open(dataset_pickle_file, 'rb') as f:
        dataset_dict = pickle.load(f)
    dataset_json_file = folder_addr / 'e2e' / 'datasets' / main_ds_name / 'config.json'
    with open(dataset_json_file, 'r') as f:
        main_dataset_config = json.load(f)

    sub_dataset_size = dataset_config["dataset_size_max"]
    assert sub_dataset_size <= dataset_size, "Sub dataset size must be less than or equal to the main dataset size"
    window_type = dataset_config["window_config"]["type"]
    history_len = dataset_config["window_config"]["history"]
    prediction_len = dataset_config["window_config"]["prediction"]
    only_arrivals = dataset_config["only_arrivals"]
    logger.info(f"Creating sub dataset with size {sub_dataset_size}, history_len: {history_len}, prediction_len: {prediction_len}, only_arrivals: {only_arrivals}")

    training_dataset = []
    max_sequence_len = 0
    max_tgt_seq_len = 0
    max_src_seq_len = 0
    dim_process = 0
    for db_dataset_dict in dataset_dict:
        db_id = db_dataset_dict['db_id']
        db_name = db_dataset_dict['dataset_name']
        db_dataset = db_dataset_dict['dataset']
        # filter broken databases
        if db_name in [ 'database_s60.db', 'database_s64.db' ]:
            continue
        logger.info(f"Processing database {db_id}, dataset size: {db_dataset_dict['size']}, arrivals number: {db_dataset_dict['arrivals_num']}")

        # apply windowing to the dataset
        if window_type == "event":
            training_db_dataset, num_arrivals, max_num_segments = window_fulltf_dataset_events(db_dataset, only_arrivals, history_len, prediction_len)
        else:
            training_db_dataset, num_arrivals, max_num_segments = window_fulltf_dataset_time(db_dataset, only_arrivals, history_len, prediction_len)
        logger.info(f"Maximum number of segments for db_id {db_id}: {max_num_segments}")
        dim_process = max(dim_process, max_num_segments+1) # arrivals and segment attempts
        db_dataset_size = len(training_db_dataset)
        db_max_seq_len = max([len(sequence) for sequence in training_db_dataset])
        db_max_tgt_seq_len = max([sum([int(event['label_mask']) for event in sequence]) for sequence in training_db_dataset])
        db_max_src_seq_len = max([sum([int(event['label_mask'] == 0) for event in sequence]) for sequence in training_db_dataset])
        max_sequence_len = max(max_sequence_len,db_max_seq_len )
        max_tgt_seq_len = max(max_tgt_seq_len,db_max_tgt_seq_len )
        max_src_seq_len = max(max_src_seq_len,db_max_src_seq_len )
        logger.info(f"Processed database {db_id}, with name: {db_name}, with size {db_dataset_size}, and found arrivals num {num_arrivals}, max seq length: {db_max_seq_len}")
        training_dataset.extend(training_db_dataset)

    dataset_size = len(training_dataset)
    logger.info(f"Total training dataset size: {dataset_size}, saving {sub_dataset_size} random entries with split ratios {split_ratios}")
    # give sub_dataset_size random numbers between 0 and dataset_size-1, they should not repeat.
    random_indices = random.sample(range(dataset_size), sub_dataset_size)
    random.shuffle(random_indices)
    sub_dataset = [training_dataset[i] for i in random_indices]

    # postprocess the absolute timestamps
    for sequence in sub_dataset:
        first_timestamp = 0
        for idx, event in enumerate(sequence):
            event['idx_event'] = idx
            if idx > 0:
                event['time_since_start'] = (event['timestamp'] - first_timestamp)*1000
            else:
                event['time_since_start'] = 0
                first_timestamp = event['timestamp']
            if event['segment'] == -1:
                # arrival event
                event['time_since_last_event'] = (event['depart_timestamp'] - event['timestamp']) * 1000

            event['type_event'] = event['segment'] + 1

    # split
    train_num = int(len(sub_dataset)*split_ratios[0])
    dev_num = int(len(sub_dataset)*split_ratios[1])
    test_num = len(sub_dataset)-train_num-dev_num
    print("train: ", train_num, " - val: ", dev_num, " - test ", test_num)

    # prepare the results folder
    results_folder_addr = folder_addr / 'e2e' / 'datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    dataset_config['dim_process'] = int(dim_process)
    # Save the dataset config
    output_config = {
        "max_sequence_len" : max_sequence_len,
        "max_tgt_seq_len": max_tgt_seq_len,
        "max_src_seq_len": max_src_seq_len,
        "train_size" : train_num,
        "val_size" : dev_num,
        "test_size" : test_num,
        "sub_size": len(sub_dataset),
        "dim_process" : int(dim_process),
        **dataset_config,
    }
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(output_config, indent=4)
        f.write(json_obj)

    # train
    train_ds = {
        'dim_process' : int(dim_process),
        'train' : sub_dataset[0:train_num],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'train.pkl', 'wb') as f:
        pickle.dump(train_ds, f)

    # dev
    dev_ds = {
        'dim_process' : int(dim_process),
        'dev' : sub_dataset[train_num:train_num+dev_num],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'dev.pkl', 'wb') as f:
        pickle.dump(dev_ds, f)

    # test
    test_ds = {
        'dim_process' : int(dim_process),
        'test' : sub_dataset[train_num+dev_num:-1],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'test.pkl', 'wb') as f:
        pickle.dump(test_ds, f)

    return


def create_fulltf_training_dataset(args):
    """
    Create a training dataset
    """

    # read configuration from args.config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    # select the source configuration
    dataset_config = dataset_config[args.configname]

    # here we create a main dataset
    # its name should be the same as the name in the config
    assert dataset_config['main_ds_name'] == args.name

    # read experiment configuration
    folder_addr = Path(args.source)
    # find all .db files in the folder
    db_files = list(folder_addr.glob("*.db"))
    if not db_files:
        logger.error("No database files found in the specified folder.")
        return
    result_database_files = [str(db_file) for db_file in db_files]
    logger.info(f"Found {len(result_database_files)} database files in the folder: {result_database_files}")

    # read exp configuration from args.config
    with open(folder_addr / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)

    time_masks = dataset_config['time_masks']

    # prepare the results folder
    results_folder_addr = folder_addr / 'e2e' / 'datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    dataset = []
    stream_rntis = []
    total_arrivals_num = 0
    total_dataset_size = 0
    dim_process = 0
    db_id = 0
    for time_mask_entry in time_masks:
        result_database_file = folder_addr / str(time_mask_entry[0])
        if str(result_database_file) not in result_database_files:
            logger.warning(f"Database {result_database_file} not found in the folder, skipping...")
            continue
        time_mask = time_mask_entry[1]
        logger.info(f"Processing database {result_database_file}")
        packet_analyzer = ULPacketAnalyzer(result_database_file)
        sched_analyzer = ULSchedulingAnalyzer(
            total_prbs_num = total_prbs_num, 
            symbols_per_slot = symbols_per_slot,
            slots_per_frame = num_slots_per_frame, 
            slots_duration_ms = slots_duration_ms, 
            scheduling_map_num_integers = scheduling_map_num_integers,
            max_num_frames = max_num_frames,
            db_addr = result_database_file
        )
        experiment_length_ts = packet_analyzer.last_ueip_ts - packet_analyzer.first_ueip_ts
        begin_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0]
        end_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1]
        logger.info(f"Database {db_id}, experiment duration: {(experiment_length_ts)} seconds")
        logger.info(f"Database {db_id}, filtering packets from {begin_ts} to {end_ts}, length: {end_ts-begin_ts} seconds")

        packets = packet_analyzer.figure_packettx_from_ts(begin_ts, begin_ts+1.0) # just take one second of packets
        packets_rnti_set = set([item['rlc.attempts'][0]['rnti'] for item in packets])
        packets_rnti_set.discard(None)
        if len(packets_rnti_set) > 1:
            logger.error("Multiple RNTIs in the packet stream, exiting...")
            return
        stream_rnti = list(packets_rnti_set)[0]
        stream_rntis.append(stream_rnti)

        dataset_this_db = extract_scheduling_events(packet_analyzer, sched_analyzer, begin_ts, end_ts, exp_config)

        # print length of dataset
        # calc number of packet arrivals only
        arrivals_num = len([item for item in dataset_this_db if item['segment'] == -1])
        total_arrivals_num += arrivals_num
        total_dataset_size += len(dataset_this_db)
        logger.info(f"Number of total events produced by db {db_id} dataset: {len(dataset_this_db)}, number of packet arrivals: {arrivals_num}")

        # append elements of one_db_dataset to dataset
        dataset.append(
            {
                'db_id' : db_id,
                'dataset_name' : str(time_mask_entry[0]),
                'stream_rnti' : stream_rnti,
                'size' : len(dataset_this_db),
                'arrivals_num' : arrivals_num,
                'dataset' : dataset_this_db
            },
        )

        db_id += 1

    logger.success(f"Number of total entries in the dataset: {total_dataset_size}, arrivals number: {total_arrivals_num}")

    # Save the dataset config
    dataset_config = {
        "stream_rntis" : stream_rntis,
        "dim_process" : int(dim_process),
        "size": total_dataset_size,
        "total_arrivals_num": total_arrivals_num,
        **dataset_config,
    }
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(dataset_config, indent=4)
        f.write(json_obj)

    # Save the dataset in a pickle file
    with open(results_folder_addr / 'dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)