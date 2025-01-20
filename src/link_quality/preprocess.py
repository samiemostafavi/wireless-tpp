import os, sys, json, random, pickle
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from edaf.core.uplink.analyze_channel import ULChannelAnalyzer
from edaf.core.uplink.analyze_packet import ULPacketAnalyzer
from edaf.core.uplink.analyze_scheduling import ULSchedulingAnalyzer

if not os.getenv('DEBUG'):
    logger.remove()
    logger.add(sys.stdout, level="INFO")

NUM_RBS_PADDING = 106
MRETX_PADDING = 4
RFAILED_PADDING = 2
NUM_BYTES_PADDING = -1

def plot_data(args):

    if args.interarrival:
        return figure_retransmission_probabilities(args)

    # read configuration from args.config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    # select the source configuration
    dataset_config = dataset_config[args.configname]

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

    time_masks = dataset_config['time_masks']
    filter_packet_sizes = dataset_config['filter_packet_sizes']
    window_config = dataset_config['window_config']
    dataset_size_max = dataset_config['dataset_size_max']
    split_ratios = dataset_config['split_ratios']
    dtime_max = dataset_config['dtime_max']
    
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    # prepare the results folder
    results_folder_addr = folder_addr / 'link_quality'/ 'pre_plots' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(dataset_config, indent=4)
        f.write(json_obj)

    # common
    arrivals_ts_list, arrivals_size_list = np.array([]), np.array([])
    mcs_val_list, mcs_ts_list = np.array([]), np.array([])

    # fast mode
    repeated_ue_rlc_val_list, repeated_ue_rlc_ts_list = np.array([]), np.array([])
    ue_ndi0_mac_val_list, ue_ndi0_mac_text_list, ue_ndi0_mac_ts_list = np.array([]), np.array([]), np.array([])

    prev_end_ts = 0
    for result_database_file, time_mask in zip(result_database_files, time_masks):
        # initiate the analyzers
        chan_analyzer = ULChannelAnalyzer(result_database_file)
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
        experiment_length_ts = chan_analyzer.last_ts - chan_analyzer.first_ts
        logger.info(f"Total experiment duration: {(experiment_length_ts)} seconds")

        begin_ts = chan_analyzer.first_ts+experiment_length_ts*time_mask[0]
        end_ts = chan_analyzer.first_ts+experiment_length_ts*time_mask[1]
        logger.info(f"Filtering link events from {begin_ts} to {end_ts}, duration: {experiment_length_ts*time_mask[1]-experiment_length_ts*time_mask[0]} seconds")

        # find the packet arrivals
        packet_arrivals = packet_analyzer.figure_packet_arrivals_from_ts(begin_ts, end_ts)
        logger.info(f"Number of packet arrivals for this duration: {len(packet_arrivals)}")
        arrivals_size_list = np.concatenate((arrivals_size_list, np.array([item['ip.in.length'] for item in packet_arrivals])))
        arrivals_ts_list = np.concatenate((arrivals_ts_list, np.array([(item['ip.in.timestamp']-begin_ts+prev_end_ts)*1000 for item in packet_arrivals])))

        # find the RNTI of the stream
        packets = packet_analyzer.figure_packettx_from_ts(begin_ts, begin_ts+1.0) # just take one second of packets
        packets_rnti_set = set([item['rlc.attempts'][0]['rnti'] for item in packets])
        # remove None from the set
        packets_rnti_set.discard(None)
        logger.info(f"RNTIs in the packet stream: {packets_rnti_set}")
        if len(packets_rnti_set) > 1:
            logger.error("Multiple RNTIs in the packet stream, exiting...")
            return
        stream_rnti = list(packets_rnti_set)[0]

        # extract MCS value time series
        mcs_arr = chan_analyzer.find_mcs_from_ts(begin_ts,end_ts)
        set_rnti = set([item['rnti'] for item in mcs_arr])
        logger.info(f"Number of unique RNTIs in MCS indices: {len(set_rnti)}")
        # filter out the MCS values for the stream RNTI
        mcs_val_list = np.concatenate((mcs_val_list, np.array([item['mcs'] for item in mcs_arr if item['rnti'] == stream_rnti])))
        mcs_ts_list = np.concatenate((mcs_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in mcs_arr if item['rnti'] == stream_rnti])))


        # find repeated RLC attempts
        repeated_ue_rlc_attempts = chan_analyzer.find_repeated_ue_rlc_attempts_from_ts(begin_ts, end_ts)
        repeated_ue_rlc_val_list = np.concatenate((repeated_ue_rlc_val_list, np.array([0 for _ in repeated_ue_rlc_attempts])))
        repeated_ue_rlc_ts_list = np.concatenate((repeated_ue_rlc_ts_list, np.array([(item['rlc.txpdu.timestamp']-begin_ts+prev_end_ts)*1000 for item in repeated_ue_rlc_attempts])))

        # find MAC attempts with ndi=0 (NACKs basically)
        ue_ndi0_mac_attempts = chan_analyzer.find_ndi0_ue_mac_attempts_from_ts(begin_ts, end_ts)
        ue_ndi0_mac_val_list = np.concatenate((ue_ndi0_mac_val_list, np.array([item['phy.tx.real_rvi'] for item in ue_ndi0_mac_attempts])))
        ue_ndi0_mac_text_list = np.concatenate((ue_ndi0_mac_text_list, np.array([item['mac.harq.hqpid'] for item in ue_ndi0_mac_attempts])))
        ue_ndi0_mac_ts_list = np.concatenate((ue_ndi0_mac_ts_list, np.array([(item['phy.tx.timestamp']-begin_ts+prev_end_ts)*1000 for item in ue_ndi0_mac_attempts])))


    # Create a subplot figure with 2 rows
    fig = make_subplots(rows=2, cols=1, subplot_titles=('MCS Index', 'Packet arrivals'))
    fig.add_trace(go.Scatter(x=mcs_ts_list, y=mcs_val_list, mode='lines+markers', name='MCS value', marker=dict(symbol='circle')), row=1, col=1)
    fig.add_trace(go.Scatter(x=arrivals_ts_list, y=arrivals_size_list, mode='markers', name='Packet arrivals', marker=dict(symbol='square')), row=2, col=1)

    # for failed_ue_rlc attempts:
    fig.add_trace(go.Scatter(x=repeated_ue_rlc_ts_list, y=repeated_ue_rlc_val_list-0.5, mode='markers', name='Repeated RLC attempts', marker=dict(symbol='triangle-down')), row=1, col=1)
    
    # for ue_ndi0_mac_val_list:
    fig.add_trace(go.Scatter(x=ue_ndi0_mac_ts_list, y=ue_ndi0_mac_val_list-0.3, mode='markers+text', name='Ue mac ndi0', marker=dict(symbol='triangle-up'), text=ue_ndi0_mac_text_list, textposition='top center'), row=1, col=1)

    fig.update_layout(
        title='Link Data Plots',
        xaxis_title='Time [ms]',
        yaxis_title='Values',
        legend_title='Legend',
    )
    fig.update_xaxes(title_text='Time [ms]', row=1, col=1)
    fig.update_yaxes(title_text='Values', row=1, col=1)
    fig.update_xaxes(title_text='Time [ms]', row=2, col=1)
    fig.update_yaxes(title_text='Values', row=2, col=1)
    fig.update_xaxes(matches='x')
    fig.write_html(str(results_folder_addr / 'fast_plot.html'))



def figure_retransmission_probabilities(args):

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
    dataset_pickle_file = folder_addr / 'link_quality' / 'datasets' / main_ds_name / 'dataset.pkl'
    with open(dataset_pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    dataset_json_file = folder_addr / 'link_quality' / 'datasets' / main_ds_name / 'config.json'
    with open(dataset_json_file, 'r') as f:
        main_dataset_config = json.load(f)

    # statistics dictionary
    stats_dict = {}

    # data lists
    prev_end_ts = 0
    seg_retx_num_list = []
    seg_failed_list = []
    seg_mcs_list = []
    seg_num_rbs_list = []
    seg_num_bytes_list = []

    for idx, sequence in enumerate(dataset):
        print(f"\rProcessing sevent {idx + 1}/{len(dataset)} ({(idx + 1) / len(dataset) * 100:.2f}%)", end="")
        event = sequence[-1]
        if event['type_event'] != 0:
            continue
        if event['num_rbs'] == NUM_RBS_PADDING:
            logger.warning(f"Padding value for num_rbs in sequence {idx}")
            continue
        t_mcs = event['mcs_index']
        if t_mcs == 0:
            t_mcs = prev_mcs
        prev_mcs = t_mcs

        t_rbs = event['num_rbs']
        t_bytes = event['num_bytes']
        t_num_retx = event['mretx']
        t_failed = event['rfailed']
        
        seg_num_rbs_list.append(t_rbs)
        seg_retx_num_list.append(t_num_retx)
        seg_failed_list.append(t_failed)
        seg_mcs_list.append(t_mcs)
        seg_num_bytes_list.append(t_bytes)

        if t_mcs not in stats_dict:
            stats_dict[t_mcs] = {
                t_rbs: {
                    'bytes': {},
                    'retx': [0,0,0,0],
                    'failed': 0,
                    'total': 0
                }
            }
        else:
            if t_rbs not in stats_dict[t_mcs]:
                stats_dict[t_mcs][t_rbs] = {
                    'bytes': {},
                    'retx': [0,0,0,0],
                    'failed': 0,
                    'total': 0
                }

        if t_bytes not in stats_dict[t_mcs][t_rbs]['bytes']:
            stats_dict[t_mcs][t_rbs]['bytes'][t_bytes] = 1
        else:
            stats_dict[t_mcs][t_rbs]['bytes'][t_bytes] += 1

        stats_dict[t_mcs][t_rbs]['retx'][t_num_retx] += 1
        stats_dict[t_mcs][t_rbs]['failed'] += t_failed
        stats_dict[t_mcs][t_rbs]['total'] += 1

    print("\n", end="")
    print(stats_dict)

    results_folder_addr = folder_addr / 'link_quality' / 'datasets' / main_ds_name
    with open(results_folder_addr / 'retx_stats.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)


def create_training_subdataset(args):
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
    dataset_pickle_file = folder_addr / 'link_quality' / 'datasets' / main_ds_name / 'dataset.pkl'
    with open(dataset_pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    dataset_json_file = folder_addr / 'link_quality' / 'datasets' / main_ds_name / 'config.json'
    with open(dataset_json_file, 'r') as f:
        main_dataset_config = json.load(f)

    dataset_size = main_dataset_config["size"]
    dataset_dim_process = main_dataset_config["dim_process"]
    new_history_len = dataset_config["window_config"]["size"]

    sub_dataset_size = dataset_config["dataset_size_max"]
    assert sub_dataset_size <= dataset_size, "Sub dataset size must be less than or equal to the main dataset size"
    assert new_history_len <= main_dataset_config["window_config"]["size"], "New history length must be less than or equal to the main dataset history length"

    # give sub_dataset_size random numbers between 0 and dataset_size-1, they should not repeat.
    random_indices = random.sample(range(dataset_size), sub_dataset_size)
    random.shuffle(random_indices)
    sub_dataset = [dataset[i][-1-new_history_len:] for i in random_indices]
    logger.info(f"Prepared sub dataset with size {len(sub_dataset)}, saving with split ratios {split_ratios}")

    # postprocess the absolute timestamps
    for sequence in sub_dataset:
        prev_time_since_start = sequence[0]['time_since_start']
        for idx, event in enumerate(sequence):
            event['idx_event'] = idx
            if idx > 0:
                event['time_since_start'] = event['time_since_last_event'] + prev_time_since_start
                prev_time_since_start = event['time_since_start']

    # split
    train_num = int(len(sub_dataset)*split_ratios[0])
    dev_num = int(len(sub_dataset)*split_ratios[1])
    test_num = len(sub_dataset)-train_num-dev_num
    print("train: ", train_num, " - val: ", dev_num, " - test ", test_num)

    # prepare the results folder
    results_folder_addr = folder_addr / 'link_quality' / 'datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)

    # Save the dataset config
    output_config = {
        "train_size" : train_num,
        "val_size" : dev_num,
        "test_size" : test_num,
        "sub_size": len(sub_dataset),
        **main_dataset_config,
    }
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(output_config, indent=4)
        f.write(json_obj)

    # train
    train_ds = {
        'dim_process' : int(dataset_dim_process),
        'train' : sub_dataset[0:train_num],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'train.pkl', 'wb') as f:
        pickle.dump(train_ds, f)

    # dev
    dev_ds = {
        'dim_process' : int(dataset_dim_process),
        'dev' : sub_dataset[train_num:train_num+dev_num],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'dev.pkl', 'wb') as f:
        pickle.dump(dev_ds, f)

    # test
    test_ds = {
        'dim_process' : int(dataset_dim_process),
        'test' : sub_dataset[train_num+dev_num:-1],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'test.pkl', 'wb') as f:
        pickle.dump(test_ds, f)

    return


def create_training_dataset(args):
    """
    Create a training dataset
    """

    # read configuration from args.config
    with open(args.config, 'r') as f:
        dataset_config = json.load(f)
    # select the source configuration
    dataset_config = dataset_config[args.configname]

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

    time_masks = dataset_config['time_masks']
    filter_successful_attempts = dataset_config['filter_successful_attempts']
    mcs_event_type = dataset_config.get('mcs_event_type', None)
    mcs_eval_interval_ms = dataset_config.get('mcs_eval_interval_ms', 100)

    # prepare the results folder
    results_folder_addr = folder_addr / 'link_quality' / 'datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)

    # here we create a main dataset
    # its name should be the same as the name in the config
    assert dataset_config['main_ds_name'] == args.name

    # select the source configuration
    window_config = dataset_config['window_config']
    split_ratios = dataset_config['split_ratios']
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    # decide about dimensions first and time_bounds
    # determine time_bounds, and stream_rntis
    db_id = 0
    dataset = []
    for result_database_file, time_mask in zip(result_database_files, time_masks):
        chan_analyzer = ULChannelAnalyzer(result_database_file)
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

        # figure time bounds
        experiment_length_ts = packet_analyzer.last_ueip_ts - packet_analyzer.first_ueip_ts
        begin_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0]
        end_ts = packet_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1]
        logger.info(f"Filtering packet arrival events from {begin_ts} to {end_ts}, duration: {experiment_length_ts*time_mask[1]-experiment_length_ts*time_mask[0]} seconds")
        logger.info(f"Database id: {db_id}, experiment duration: {(experiment_length_ts)} seconds")
        logger.info(f"Database id: {db_id}, filtering link events from {begin_ts} to {end_ts}, duration: {end_ts-begin_ts} seconds")

        # figure stream rnti
        packets = packet_analyzer.figure_packettx_from_ts(begin_ts, begin_ts+1.0) # just take one second of packets
        packets_rnti_set = set([item['rlc.attempts'][0]['rnti'] for item in packets])
        packets_rnti_set.discard(None)
        if len(packets_rnti_set) > 1:
            logger.error("Multiple RNTIs in the packet stream, exiting...")
            return
        stream_rnti = list(packets_rnti_set)[0]

        # if window config is block_event, then don't extract mcs_events
        mcs_event_type = None if window_config['type'] == 'block_event' else mcs_event_type

        # extract events
        segment_events, mcs_events = extract_link_quality_events(chan_analyzer, packet_analyzer, sched_analyzer, stream_rnti, begin_ts, end_ts, exp_config, mcs_event_type, mcs_eval_interval_ms)

        if filter_successful_attempts:
            segment_events = [item for item in segment_events if (item['mretx'] > 0 or item['rfailed'])]

        link_events = [ *segment_events, *mcs_events ]
        link_events = sorted(link_events, key=lambda x: x['timestamp'], reverse=False)

        history_window_size = int(dataset_config['window_config']['size'])
        dataset_size_max = int(dataset_config['dataset_size_max'])

        if window_config['type'] == 'segment_event':
            dataset_this_db = []
            for l in range(len(segment_events) - 2, -1, -1):
                if l <= history_window_size-1:
                    break
                l1 = l + 1
                hist_sequence = window_history_segment_events(l1, segment_events, history_window_size)
                if len(hist_sequence) > 0:
                    dataset_this_db.append(hist_sequence)
                if len(dataset_this_db) > dataset_size_max:
                    break

        elif window_config['type'] == 'mcs_event':
            if mcs_event_type == 'change':
                dataset_this_db = dataset_create_mcs_change(link_events, dataset_config)
            elif mcs_event_type == 'decision':
                dataset_this_db = []
                for k in range(len(link_events) - 2, -1, -1):
                    if k <= history_window_size-1:
                        break
                    k1 = k + 1
                    # the label event should be an mcs event
                    if link_events[k1]['type_event'] == 0:
                        continue
                    hist_sequence = window_history_mcs_decision_events(k1, link_events, history_window_size)
                    if len(hist_sequence) > 0:
                        dataset_this_db.append(hist_sequence)
                    if len(dataset_this_db) > dataset_size_max:
                        break
            else:
                logger.error("Invalid mcs event type")
                return
        else:
            logger.error("Invalid window type")

        # print length of dataset
        logger.info(f"Number of total entries produced by db {db_id} dataset: {len(dataset_this_db)}")
        if len(dataset_this_db) > 0:
            print(dataset_this_db[0])

        # append elements of one_db_dataset to dataset
        dataset.extend(dataset_this_db)

        db_id += 1

    logger.success(f"Number of total entries in the dataset: {len(dataset)}")

    # event types: 0: retransmissions, 1: mcs related event
    dim_process = 2

    # Save the dataset config
    dataset_config = {
        "dim_process" : int(dim_process),
        "size": len(dataset),
        **dataset_config,
    }
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(dataset_config, indent=4)
        f.write(json_obj)

    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)


def extract_link_quality_events(chan_analyzer, packet_analyzer, sched_analyzer, stream_rnti, begin_ts, end_ts, exp_config, mcs_event_type, mcs_eval_interval_ms = 100):

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

                    
    # extract MCS events from channel
    mcs_events_arr_0 = []
    if mcs_event_type is not None:
        mcs_list = chan_analyzer.find_mcs_from_ts(begin_ts,end_ts)
        mcs_list = [item for item in mcs_list if item['rnti'] == stream_rnti]
        mcs_list.sort(key=lambda x: x['timestamp'])
    if mcs_event_type == 'change':
        # extract MCS change events
        filtered_mcs_list = []
        previous_mcs = None
        for item in mcs_list:
            if item['mcs'] != previous_mcs:
                filtered_mcs_list.append(item)
                previous_mcs = item['mcs']
        for item in filtered_mcs_list:
            frame_start_ts, frame_num, slot_num = sched_analyzer.find_frame_slot_from_ts(
                timestamp=item['timestamp'],
                SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
            )
            time_since_frame0 = frame_num*num_slots_per_frame*slots_duration_ms + slot_num*slots_duration_ms
            mcs_events_arr_0.append({
                'type_event' : 1, # MCS event
                'timestamp' : item['timestamp'],
                'time_since_start':time_since_frame0,
                'mcs_index' : item['mcs'],
                'rfailed' : RFAILED_PADDING,
                'mretx' : MRETX_PADDING,
                'num_rbs': NUM_RBS_PADDING,
                'num_bytes': NUM_BYTES_PADDING,
            })
    elif mcs_event_type == 'decision':
        # needs mcs_eval_interval_ms to be set
        # extract MCS decision events (with an interval)
        # Downsample to 100ms intervals with majority rule
        # Iterate through intervals
        for interval_start in np.arange(begin_ts, end_ts + mcs_eval_interval_ms/1000, mcs_eval_interval_ms/1000):
            interval_end = interval_start + mcs_eval_interval_ms/1000
            # Find all MCS values in the interval
            mcs_in_interval = [
                report['mcs'] for report in mcs_list if (
                    (interval_start <= report['timestamp']) and \
                        (report['timestamp'] < interval_end)
                )
            ]
            # Determine the majority MCS value
            if mcs_in_interval:
                mcs_counter = Counter(mcs_in_interval)
                majority_mcs = mcs_counter.most_common(1)[0][0]  # Most common MCS value
            else:
                majority_mcs = None  # No reports in the interval
                logger.warning(f"No MCS reports in the interval {interval_start}-{interval_end}")
                continue
            mcs_value = majority_mcs
            mcs_timestamp = interval_start

            # find mcs report's timestamp since the start of the frame
            frame_start_ts, frame_num, slot_num = sched_analyzer.find_frame_slot_from_ts(
                timestamp=mcs_timestamp,
                SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
            )
            if frame_start_ts is None:
                logger.warning(f"Could not find frame start timestamp for MCS report at {mcs_timestamp}")
                continue
            time_since_frame0 = frame_num*num_slots_per_frame*slots_duration_ms + slot_num*slots_duration_ms
            mcs_events_arr_0.append({
                'type_event' : 1, # MCS event
                'timestamp' : mcs_timestamp,
                'time_since_start':time_since_frame0,
                'mcs_index' : mcs_value,
                'rfailed' : RFAILED_PADDING,
                'mretx' : MRETX_PADDING,
                'num_rbs': NUM_RBS_PADDING,
                'num_bytes': NUM_BYTES_PADDING,
            })

    if mcs_event_type is not None:
        # sort the mcs events based on timestamp
        mcs_events_arr_0 = sorted(mcs_events_arr_0, key=lambda x: x['timestamp'], reverse=False)

    # extract block events array from the stream
    block_events_arr_0 = []
    packets = packet_analyzer.figure_packettx_from_ts(begin_ts, end_ts)
    prev_mcs = 0
    for idx, packet in enumerate(packets):
        print(f"\rProcessing packet {idx + 1}/{len(packets)} ({(idx + 1) / len(packets) * 100:.2f}%) with packet sn: {packet['sn']}", end="")
        for idx2, rlc_attempt in enumerate(packet['rlc.attempts']):
            mcs_index = rlc_attempt['mac.attempts'][0]['mcs']
            if mcs_index == 0:
                mcs_index = prev_mcs
            prev_mcs = mcs_index

            num_rbs = rlc_attempt['mac.attempts'][0]['rbs']
            num_bytes = rlc_attempt['len']
            mretx = len(rlc_attempt['mac.attempts'])-1
            rfailed = int(not rlc_attempt['acked'])

            frame_start_ts, frame_num, slot_num = sched_analyzer.find_frame_slot_from_ts(
                timestamp=rlc_attempt['mac.attempts'][0]['phy.in_t'],
                SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
            )
            time_since_frame0 = frame_num*num_slots_per_frame*slots_duration_ms + slot_num*slots_duration_ms
            block_events_arr_0.append(
                {
                    'type_event': 0, #block attempt event
                    'timestamp': rlc_attempt['mac.attempts'][0]['phy.in_t'],
                    'time_since_start': time_since_frame0,
                    'mcs_index': mcs_index,
                    'rfailed': rfailed,
                    'mretx': mretx,
                    'num_rbs': num_rbs,
                    'num_bytes' : num_bytes
                }
            )
    print("\n", end="")

    logger.info(f"This db, number of block events: {len(block_events_arr_0)}, number of MCS events: {len(mcs_events_arr_0)}")

    return block_events_arr_0, mcs_events_arr_0


def window_history_segment_events(l1, segment_events, history_window_size):
    events_window = []
    prev_event_ts = 0
    if l1 - history_window_size < 0 or l1 + 1 > len(segment_events):
        return []
    for pos, event in enumerate(segment_events[l1-history_window_size:l1+1]):
        events_window.append(
            {
                'idx_event' : pos, 
                'time_since_last_event' : (event['timestamp'] - prev_event_ts)*1000 if pos > 0 else 0,
                **event
            }
        )
        prev_event_ts = event['timestamp']
    return events_window

def window_history_mcs_decision_events(k1, link_events, history_window_size):
    events_window = []
    prev_event_ts = 0
    if ((k1 - history_window_size) < 0) or ((k1 + 1) > len(link_events)):
        return []
    pos = 0
    for event in link_events[k1-history_window_size:k1+1]:
        events_window.append(
            {
                'idx_event' : pos, 
                'time_since_last_event' : (event['timestamp'] - prev_event_ts)*1000 if pos > 0 else 0,
                **event
            }
        )
        pos += 1
        prev_event_ts = event['timestamp']
    return events_window


def dataset_create_mcs_change(sorted_link_events, dataset_config):

    # select the source configuration
    window_config = dataset_config['window_config']
    history_window_size = window_config['size']
    dataset_size_max = dataset_config['dataset_size_max']

    dataset = []
    for idx,_ in enumerate(sorted_link_events):
        if idx+history_window_size >= len(sorted_link_events):
            break

        # create the history sequence, with the size of history_window_size - 1
        events_window = []
        prev_event_ts = 0
        for pos, event in enumerate(sorted_link_events[idx:idx+history_window_size-1]):
            events_window.append(
                {
                    'idx_event' : pos, 
                    'time_since_last_event' : (event['timestamp'] - prev_event_ts)*1000 if pos > 0 else 0,
                    **event
                }
            )
            prev_event_ts = event['timestamp']
        
        # now look for the MCS change event after the last event in the window
        mcs_event = None
        for event in sorted_link_events[idx+history_window_size-2:]:
            if event['type_event'] == 1:
                mcs_event = event
                break

        if mcs_event is None:
            logger.info(f"Could not find MCS change event after the last event in the window, breaking...")
            break
        
        events_window.append(
            {
                'idx_event' : pos+1, 
                'time_since_last_event' : (mcs_event['timestamp'] - prev_event_ts)*1000,
                **mcs_event
            }
        )
    
        #print(events)
        dataset.append(events_window)
        if len(dataset) > dataset_size_max:
            break

    return dataset