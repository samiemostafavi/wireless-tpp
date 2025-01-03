import pickle
import random
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
            arrival_history_sequence = window_history_arrival_events(n+1, packet_arrival_events, arrival_dataset_config['window_config']['size'])
            if len(arrival_history_sequence) == 0:
                continue

            # find arrival_event in scheduling_events
            sched_event_m, m = find_event(arrival_event_n, scheduling_events, {'key': 'segment', 'value': -1})
            sched_event_mL, mL = find_event(arrival_event_nL, scheduling_events, {'key': 'segment', 'value': -1})
            if sched_event_m is None or sched_event_mL is None:
                continue

            # first segment of packet n
            sched_event_m1 = scheduling_events[m+1]
            sched_history_sequence = window_history_scheduling_events(m+1, scheduling_events, scheduling_dataset_config['window_config']['size'])
            if len(sched_history_sequence) == 0:
                continue

            # find it in link_segment_events
            segment_event_l1, l1 = find_event(sched_event_m1, link_segment_events)
            if segment_event_l1 is None:
                continue
            retx_history_sequence = window_history_segment_events(l1, link_segment_events, retx_dataset_config['window_config']['size'])
            if len(retx_history_sequence) == 0:
                continue

            # find the label mcs event
            link_event_mcs_k1, mcs_k1 = find_next_event_after_input(sched_event_m, filtered_link_events, {'key': 'type_event', 'value': 1})
            if link_event_mcs_k1 is None:
                continue
            mcs_history_sequence = window_history_mcs_decision_events(mcs_k1, filtered_link_events, mcs_dataset_config['window_config']['size'], sched_event_m['timestamp'] + 0.0001)
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

    
