import os, sys, json, random, pickle
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from loguru import logger

from edaf.core.uplink.analyze_packet import ULPacketAnalyzer
from edaf.core.uplink.analyze_scheduling import ULSchedulingAnalyzer

if not os.getenv('DEBUG'):
    logger.remove()
    logger.add(sys.stdout, level="INFO")

def plot_data(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = config[args.configname]
    
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

    time_masks = config['time_masks']
    filter_packet_sizes = config['filter_packet_sizes']
    window_config = config['window_config']
    dataset_size_max = config['dataset_size_max']
    split_ratios = config['split_ratios']
    dtime_max = config['dtime_max']
    
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']

    # prepare the results folder
    results_folder_addr = folder_addr / 'packet_arrival' / 'pre_plots' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(config, indent=4)
        f.write(json_obj)

    packet_size_arr, packet_inp_ts_arr, frame_strt_ts_arr, slot_num_arr, delta_times_ms = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    prev_end_ts = 0
    for result_database_file, time_mask in zip(result_database_files, time_masks):

        # initiate analyzers
        pacekt_analyzer = ULPacketAnalyzer(result_database_file)
        scheduling_analyzer = ULSchedulingAnalyzer(
            total_prbs_num = total_prbs_num, 
            symbols_per_slot = symbols_per_slot,
            slots_per_frame = num_slots_per_frame, 
            slots_duration_ms = slots_duration_ms, 
            scheduling_map_num_integers = scheduling_map_num_integers,
            max_num_frames = max_num_frames,
            db_addr = result_database_file
        )
        end_ts = pacekt_analyzer.last_ueip_ts
        begin_ts = pacekt_analyzer.first_ueip_ts
        experiment_length_ts = pacekt_analyzer.last_ueip_ts - pacekt_analyzer.first_ueip_ts
        logger.info(f"Experiment duration: {(experiment_length_ts)} seconds")
        logger.info(f"Filtering packets from {pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0]} to {pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1]}, length: {experiment_length_ts*time_mask[1]-experiment_length_ts*time_mask[0]} seconds")
        packets = pacekt_analyzer.figure_packet_arrivals_from_ts(pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[0], pacekt_analyzer.first_ueip_ts+experiment_length_ts*time_mask[1])
        logger.info(f"Number of packets for this duration: {len(packets)}")

        # sort the packets in case they are not sorted
        packets = sorted(packets, key=lambda x: x['ip.in.timestamp'], reverse=False)

        packet_size_arr_this_db = []
        packet_inp_ts_arr_this_db = []
        frame_strt_ts_arr_this_db = []
        slot_num_arr_this_db = []
        delta_times_ms_this_db = []
        prev_arrival_ts = 0
        for packet in packets:
            if len(filter_packet_sizes) > 0 and packet['ip.in.length'] not in filter_packet_sizes:
                continue
            frame_start_ts, frame_num, slot_num = scheduling_analyzer.find_frame_slot_from_ts(
                timestamp=packet['ip.in.timestamp'], 
                #SCHED_OFFSET_S=0.002 # 2ms which is 4*slot_duration_ms
                #SCHED_OFFSET_S=0.004 # 4ms which is 8*slot_duration_ms
                SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
            )
            if prev_arrival_ts is not 0:
                delta_times_ms_this_db.append((packet['ip.in.timestamp']-prev_arrival_ts+prev_end_ts)*1000)
            prev_arrival_ts = packet['ip.in.timestamp']
            slot_num_arr_this_db.append(slot_num)
            frame_strt_ts_arr_this_db.append(frame_start_ts)
            packet_inp_ts_arr_this_db.append(packet['ip.in.timestamp'])
            packet_size_arr_this_db.append(packet['ip.in.length'])

        packet_size_arr = np.concatenate((packet_size_arr, np.array(packet_size_arr_this_db)))
        packet_inp_ts_arr = np.concatenate((packet_inp_ts_arr, np.array(packet_inp_ts_arr_this_db)))
        frame_strt_ts_arr = np.concatenate((frame_strt_ts_arr, np.array(frame_strt_ts_arr_this_db)))
        slot_num_arr = np.concatenate((slot_num_arr, np.array(slot_num_arr_this_db)))
        delta_times_ms = np.concatenate((delta_times_ms, np.array(delta_times_ms_this_db)))
        prev_end_ts = (end_ts-begin_ts) + prev_end_ts

    # Plot probability distribution of delta times
    import plotly.express as px
    # Create a histogram plot of the delta times
    fig = px.histogram(x=delta_times_ms, nbins=100, title='Probability Distribution of Delta Times', histnorm='probability')
    fig.update_layout(xaxis_range=[0, dtime_max])
    fig.update_layout(xaxis_title='Delta Time (ms)', yaxis_title='Probability')
    fig.write_html(str(results_folder_addr / 'delta_times.html'))

    # Plot timeseries of slot numbers against packet_inp_ts
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=packet_inp_ts_arr, y=slot_num_arr, mode='markers', name='Slot Numbers'))
    fig1.add_trace(go.Scatter(x=frame_strt_ts_arr, y=np.ones(len(frame_strt_ts_arr)), mode='markers', name='Frame Start'))
    fig1.update_layout(title='Timeseries of packet arrival events', xaxis_title='Time', yaxis_title='Slot Number')
    fig1.write_html(str(results_folder_addr / 'slot_numbers.html'))

    # Plot timeseries of (packet_inp_ts-framest_ts_arr)*1000 against packet_inp_ts
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=packet_inp_ts_arr, y=(np.array(packet_inp_ts_arr)-np.array(frame_strt_ts_arr))*1000, mode='markers', name='Time Difference'))
    fig2.update_layout(title='Timeseries of (packet_inp_ts-framest_ts_arr)*1000', xaxis_title='Packet Input Time', yaxis_title='Time Difference (ms)')
    fig2.write_html(str(results_folder_addr / 'time_difference.html'))

    # Plot timeseries of packet_size_arr
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=packet_inp_ts_arr, y=packet_size_arr, mode='markers', name='Packet Sizes'))
    fig3.update_layout(title='Timeseries of packet sizes', xaxis_title='Packet Input Time', yaxis_title='Packet Sizes (bytes)')
    fig3.write_html(str(results_folder_addr / 'packet_sizes.html'))

    
def create_training_dataset(args):

    # read configuration from args.config
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = config[args.configname]

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

    time_masks = config['time_masks']
    filter_packet_sizes = config['filter_packet_sizes']
    window_config = config['window_config']
    dataset_size_max = config['dataset_size_max']
    split_ratios = config['split_ratios']
    dtime_max = config['dtime_max']

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']

    # Save configuration dictionary to a json file
    results_folder_addr = folder_addr / 'packet_arrival'  / 'datasets' / args.name
    results_folder_addr.mkdir(parents=True, exist_ok=True)
    with open(results_folder_addr / 'config.json', 'w') as f:
        json_obj = json.dumps(config, indent=4)
        f.write(json_obj)


    history_window_size = window_config['size']

    packet_sizes_set = set()
    dataset = []
    for result_database_file, time_mask in zip(result_database_files, time_masks):

        # initiate analyzers
        pacekt_analyzer = ULPacketAnalyzer(result_database_file)
        scheduling_analyzer = ULSchedulingAnalyzer(
            total_prbs_num = total_prbs_num, 
            symbols_per_slot = symbols_per_slot,
            slots_per_frame = num_slots_per_frame, 
            slots_duration_ms = slots_duration_ms, 
            scheduling_map_num_integers = scheduling_map_num_integers,
            max_num_frames = max_num_frames,
            db_addr = result_database_file
        )
        end_ts = pacekt_analyzer.last_ueip_ts
        begin_ts = pacekt_analyzer.first_ueip_ts
        experiment_length_ts = end_ts - begin_ts
        logger.info(f"Experiment duration: {(experiment_length_ts)} seconds")
        logger.info(f"Filtering packets from {begin_ts+experiment_length_ts*time_mask[0]} to {begin_ts+experiment_length_ts*time_mask[1]}, length: {experiment_length_ts*time_mask[1]-experiment_length_ts*time_mask[0]} seconds")
        packets = pacekt_analyzer.figure_packettx_from_ts(begin_ts+experiment_length_ts*time_mask[0], begin_ts+experiment_length_ts*time_mask[1])
        logger.info(f"Number of packets for this duration: {len(packets)}")

        # sort the packets in case they are not sorted
        packets = sorted(packets, key=lambda x: x['ip.in_t'], reverse=False)

        packet_arrival_events = []
        last_event_ts = 0
        for packet in packets:
            if len(filter_packet_sizes) > 0 and packet['len'] not in filter_packet_sizes:
                continue
            packet_sizes_set.add(int(packet['len']))
            frame_start_ts, frame_num, slot_num = scheduling_analyzer.find_frame_slot_from_ts(
                timestamp=packet['ip.in_t'],
                #SCHED_OFFSET_S=0.002 # 2ms which is 4*slot_duration_ms
                #SCHED_OFFSET_S=0.004 # 4ms which is 8*slot_duration_ms
                SCHED_OFFSET_S=scheduling_time_ahead_ms/1000 # 4ms which is 8*slot_duration_ms
            )

            time_since_frame0 = frame_num*num_slots_per_frame*slots_duration_ms + slot_num*slots_duration_ms
            time_since_last_event = time_since_frame0-last_event_ts
            #if time_since_last_event < 0:
            #    time_since_last_event = time_since_frame0 + max_num_frames*num_slots_per_frame*slots_duration_ms
            if time_since_last_event < 0:
                time_since_last_event = time_since_frame0

            last_event_ts = time_since_frame0

            if time_since_last_event > dtime_max:
                continue

            packet_arrival_events.append(
                {
                    'type_event' : packet['len'],
                    'time_since_start' : time_since_frame0,
                    'time_since_last_event' : time_since_last_event,
                    'timestamp' : packet['ip.in_t']
                }
            )
        
        dataset_this_db = []
        for idx,_ in enumerate(packet_arrival_events):
            if idx+history_window_size >= len(packet_arrival_events):
                break
            events_window = []
            for pos, event in enumerate(packet_arrival_events[idx:idx+history_window_size]):
                idx_event = pos
                events_window.append(
                    {
                        'idx_event' : pos,
                        'type_event': event['type_event'],
                        'time_since_start' : event['time_since_start'],
                        'time_since_last_event' : event['time_since_last_event'],
                    }
                )        

            #print(events)
            dataset_this_db.append(events_window)
            if len(dataset_this_db) > dataset_size_max:
                break

        # print length of dataset
        logger.info(f"Number of total entries produced by this db dataset: {len(dataset_this_db)}")

        # append elements of one_db_dataset to dataset
        dataset.extend(dataset_this_db)

    # shuffle
    random.shuffle(dataset)

    dim_process = len(list(packet_sizes_set))
    logger.info(f"Number of unique packet sizes and dim_process: {dim_process}")

    # print length of dataset
    logger.info(f"Number of total entries in dataset: {len(dataset)}")
    print(dataset[0])

    # split
    train_num = int(len(dataset)*split_ratios[0])
    dev_num = int(len(dataset)*split_ratios[1])
    print("train: ", train_num, " - dev: ", dev_num)
    # train
    train_ds = {
        'dim_process' : dim_process,
        'train' : dataset[0:train_num],
    }

    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'train.pkl', 'wb') as f:
        pickle.dump(train_ds, f)
    # dev
    dev_ds = {
        'dim_process' : dim_process,
        'dev' : dataset[train_num:train_num+dev_num],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'dev.pkl', 'wb') as f:
        pickle.dump(dev_ds, f)
    # test
    test_ds = {
        'dim_process' : dim_process,
        'test' : dataset[train_num+dev_num:-1],
    }
    # Save the dictionary to a pickle file
    with open(results_folder_addr / 'test.pkl', 'wb') as f:
        pickle.dump(test_ds, f)