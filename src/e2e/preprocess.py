import os, sys, json
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from edaf.core.uplink.analyze_packet import ULPacketAnalyzer

from src.link_quality import extract_link_quality_events_stream_based
from src.packet_arrival import extract_packet_arrival_events
from src.scheduling import extract_scheduling_events

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

        arrivals_type_list = np.concatenate((arrivals_type_list, np.array([item['type_event'] for item in packet_arrival_events])))
        arrivals_ts_list = np.concatenate((arrivals_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in packet_arrival_events])))

        link_quality_events = [ *link_retransmission_events, *link_mcs_events ]
        sorted_link_quality_events = sorted(link_quality_events, key=lambda x: x['timestamp'], reverse=False)
        link_type_list = np.concatenate((link_type_list, np.array([item['type_event'] for item in sorted_link_quality_events])))
        link_ts_list = np.concatenate((link_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in sorted_link_quality_events])))

        scheduling_type_list = np.concatenate((scheduling_type_list, np.array([item['segment']+1 for item in scheduling_events])))
        scheduling_ts_list = np.concatenate((scheduling_ts_list, np.array([(item['timestamp']-begin_ts+prev_end_ts)*1000 for item in scheduling_events])))
        depart_ts_list = np.concatenate((depart_ts_list, np.array([(item['depart_timestamp']-begin_ts+prev_end_ts)*1000 for item in scheduling_events if item['depart_timestamp']>0])))

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
            name='History events'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=link_ts_list, 
            y=np.ones(len(link_ts_list)),
            mode='markers+text', 
            text=link_type_list,
            textposition='top center',
            name='History events'
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
            name='History events'
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
            name='History events'
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
