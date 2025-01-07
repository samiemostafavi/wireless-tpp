import plotly.graph_objects as go
from pathlib import Path
import yaml, pickle, json, copy
import numpy as np

from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerScheduling
from wireless_tpp.utils import logger

MAX_NUM_SEGMENTS = 4

def calc_delay_pred_s(pred_seq):
    arrival_ts = pred_seq[0]['timestamp']
    last_seg_ts = pred_seq[-1]['timestamp']
    departure_ts = last_seg_ts + float(pred_seq[-1]['mretx'])*0.01 # 10ms per retransmission
    return departure_ts - arrival_ts

def calc_delay_label_s(label_seq):
    arrival_ts = label_seq[0]['timestamp']
    departure_ts = label_seq[0]['depart_timestamp']
    return departure_ts - arrival_ts

def calc_errors(pred_seq_arr, label_seq):
    # all schedule's time difference plus arrival time difference
    # example:
    # label [5.1, 10.4, 15.4, 20.1] (time since packet arrival)
    # pred [4, 10.8, 13.5]
    # here we won't consider if there are more segments in the label than the prediction
    num_pred_samples = len(pred_seq_arr)
    label_delay = calc_delay_label_s(label_seq)

    errors_mretx = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))
    pred_mretx = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))

    errors_rfailed = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))
    pred_rfailed = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))

    errors_ts = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))
    pred_ts = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))

    errors_len = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))
    pred_len = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS+1))

    errors_delay = np.zeros((num_pred_samples))
    pred_delays = np.zeros((num_pred_samples))

    for idx, pred_seq in enumerate(pred_seq_arr): # idx chooses over the number of samples
        pred_delay = calc_delay_pred_s(pred_seq)
        pred_delays[idx] = pred_delay
        errors_delay[idx] = abs(label_delay - pred_delay)
        for idy in range(MAX_NUM_SEGMENTS+1):
            if len(label_seq) <= idy or len(pred_seq) <= idy:
                errors_ts[idx,idy] = np.nan
                errors_len[idx,idy] = np.nan
                errors_mretx[idx,idy] = np.nan
                errors_rfailed[idx,idy] = np.nan
            else:
                errors_ts[idx,idy] = abs(label_seq[idy]['timestamp'] - pred_seq[idy]['timestamp'])
                errors_len[idx,idy] = abs(label_seq[idy]['len'] - pred_seq[idy]['len'])
                errors_mretx[idx,idy] = abs(label_seq[idy]['mretx'] - pred_seq[idy]['mretx'])
                errors_rfailed[idx,idy] = abs(label_seq[idy]['rfailed'] - pred_seq[idy]['rfailed'])

            if len(pred_seq) <= idy:
                pred_ts[idx,idy] = np.nan
                pred_len[idx,idy] = np.nan
                pred_mretx[idx,idy] = np.nan
                pred_rfailed[idx,idy] = np.nan
            else:
                pred_ts[idx,idy] = pred_seq[idy]['timestamp']
                pred_len[idx,idy] = pred_seq[idy]['len']
                pred_mretx[idx,idy] = pred_seq[idy]['mretx']
                pred_rfailed[idx,idy] = pred_seq[idy]['rfailed']

    # take mean of errors over all the samples, except the entries with nan
    pred_delay_std = np.nanstd(pred_delays)*1000
    mean_errors_delay = np.nanmean(errors_delay)*1000

    mean_errors_ts = np.nanmean(errors_ts, axis=0)*1000
    ts_std = np.nanstd(pred_ts, axis=0)*1000

    mean_errors_len = np.nanmean(errors_len, axis=0)
    len_std = np.nanstd(pred_len, axis=0)

    mean_errors_mretx = np.nanmean(errors_mretx, axis=0)
    mretx_std = np.nanstd(pred_mretx, axis=0)

    mean_errors_rfailed = np.nanmean(errors_rfailed, axis=0)
    rfailed_std = np.nanstd(pred_rfailed, axis=0)

    return mean_errors_delay, pred_delay_std, mean_errors_ts, ts_std, mean_errors_len, len_std, mean_errors_mretx, mretx_std, mean_errors_rfailed, rfailed_std

def evaluate_model(args):

    # open the files and load the data
    log_folder = Path(args.source) / "e2e" / "prediction_results" / args.name / args.id
    conf_file = next(log_folder.glob("*.json"))
    with open(conf_file, 'r') as file:
        e2e_config = json.load(file)
    num_packets = e2e_config["num_future_packet_predictions"]

    pkl_file = next(log_folder.glob("*.pkl"))
    with open(pkl_file, 'rb') as file:
        dataset = pickle.load(file)

    num_entries = len(dataset)
    total_errors_delay = np.zeros((num_packets))
    total_pred_delay_std = np.zeros((num_packets))

    total_errors_ts = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))
    total_ts_std = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))

    total_errors_len = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))
    total_len_std = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))

    total_errors_mretx = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))
    total_mretx_std = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))

    total_errors_rfailed = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))
    total_rfailed_std = np.zeros((num_packets, MAX_NUM_SEGMENTS+1))

    count = 0
    for entry in dataset:
        # fix the label packets
        packets_label = []
        sequence = [ entry["scheduling"][-1] ]
        for event in entry["label"]:
            if event['segment'] == -1 and len(sequence) > 1:
                packets_label.append(copy.deepcopy(sequence))
                sequence = []
            if len(packets_label) >= num_packets:
                break
            sequence.append(event)

        # calc the error
        if 'pred' not in entry:
            continue
        count += 1
        for idx, packet_pred in enumerate(entry["pred"]): # idx: packet number
            mean_errors_delay, pred_delay_std, \
                mean_errors_ts, ts_std, mean_errors_len, len_std, \
                mean_errors_mretx, mretx_std, mean_errors_rfailed, rfailed_std = \
                    calc_errors(packet_pred, packets_label[idx])

            total_errors_delay[idx] += mean_errors_delay
            total_pred_delay_std[idx] += pred_delay_std

            total_errors_ts[idx,:] += mean_errors_ts
            total_ts_std[idx,:] += ts_std

            total_errors_len[idx,:] += mean_errors_len
            total_len_std[idx,:] += len_std

            total_errors_mretx[idx,:] += mean_errors_mretx
            total_mretx_std[idx,:] += mretx_std

            total_errors_rfailed[idx,:] += mean_errors_rfailed
            total_rfailed_std[idx,:] += rfailed_std
            
    total_errors_delay = total_errors_delay/count
    total_pred_delay_std = total_pred_delay_std/count

    print("Mean delay error: ", total_errors_delay)
    print("Mean delay std: ", total_pred_delay_std)

    total_errors_ts = total_errors_ts/count
    total_ts_std = total_ts_std/count

    print("Mean timestamp error: ", total_errors_ts)
    print("Mean timestamp std: ", total_ts_std)

    total_errors_len = total_errors_len/count
    total_len_std = total_len_std/count

    print("Mean length error: ", total_errors_len)
    print("Mean length std: ", total_len_std)

    total_errors_mretx = total_errors_mretx/count
    total_mretx_std = total_mretx_std/count

    print("Mean mretx error: ", total_errors_mretx)
    print("Mean mretx std: ", total_mretx_std)

    total_errors_rfailed = total_errors_rfailed/count
    total_rfailed_std = total_rfailed_std/count

    print("Mean rfailed error: ", total_errors_rfailed)
    print("Mean rfailed std: ", total_rfailed_std)

    

    

    
        
    


