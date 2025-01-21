import plotly.graph_objects as go
from pathlib import Path
import yaml, pickle, json, copy
import numpy as np

from wireless_tpp.config_factory import Config
from wireless_tpp.runner import TPPRunnerScheduling
from wireless_tpp.utils import logger
import matplotlib.pyplot as plt

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

    errors_seg_mretx = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_mretx = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))

    errors_seg_rfailed = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_rfailed = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))

    errors_seg_ts = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_ts = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))

    errors_seg_len = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_len = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))

    errors_arr_ts = np.zeros((num_pred_samples))
    pred_arr_ts = np.zeros((num_pred_samples))
    pred_arr_len = np.zeros((num_pred_samples))
    errors_arr_len = np.zeros((num_pred_samples))

    errors_delay = np.zeros((num_pred_samples))
    pred_delays = np.zeros((num_pred_samples))

    label_delay = calc_delay_label_s(label_seq)
    for idx, pred_seq in enumerate(pred_seq_arr): # idx iterates over the number of samples
        pred_delay = calc_delay_pred_s(pred_seq)
        pred_delays[idx] = pred_delay
        errors_delay[idx] = abs(label_delay - pred_delay)
        errors_arr_ts[idx] = abs(label_seq[0]['timestamp'] - pred_seq[0]['timestamp'])
        errors_arr_len[idx] = abs(label_seq[0]['len'] - pred_seq[0]['len'])
        pred_arr_ts[idx] = pred_seq[0]['timestamp']
        pred_arr_len[idx] = pred_seq[0]['len']

        for idy in range(MAX_NUM_SEGMENTS):
            if len(label_seq)-1 <= idy or len(pred_seq)-1 <= idy:
                errors_seg_mretx[idx,idy] = np.nan
                pred_seg_mretx[idx,idy] = np.nan
                errors_seg_rfailed[idx,idy] = np.nan
                pred_seg_rfailed[idx,idy] = np.nan
                errors_seg_ts[idx,idy] = np.nan
                pred_seg_ts[idx,idy] = np.nan
                errors_seg_len[idx,idy] = np.nan
                pred_seg_len[idx,idy] = np.nan
            else:
                errors_seg_ts[idx,idy] = abs(label_seq[idy+1]['timestamp'] - pred_seq[idy+1]['timestamp'])
                errors_seg_len[idx,idy] = abs(label_seq[idy+1]['len'] - pred_seq[idy+1]['len'])
                errors_seg_mretx[idx,idy] = abs(label_seq[idy+1]['mretx'] - pred_seq[idy+1]['mretx'])
                errors_seg_rfailed[idx,idy] = abs(label_seq[idy+1]['rfailed'] - pred_seq[idy+1]['rfailed'])

            if len(pred_seq)-1 <= idy:
                pred_seg_ts[idx,idy] = np.nan
                pred_seg_len[idx,idy] = np.nan
                pred_seg_mretx[idx,idy] = np.nan
                pred_seg_rfailed[idx,idy] = np.nan
            else:
                pred_seg_ts[idx,idy] = pred_seq[idy+1]['timestamp']
                pred_seg_len[idx,idy] = pred_seq[idy+1]['len']
                pred_seg_mretx[idx,idy] = pred_seq[idy+1]['mretx']
                pred_seg_rfailed[idx,idy] = pred_seq[idy+1]['rfailed']

    # take mean of errors over all the samples, except the entries with nan
    pred_delay_std = np.nanstd(pred_delays)*1000
    mean_errors_delay = np.nanmean(errors_delay)*1000

    mean_errors_arr_ts = np.nanmean(errors_arr_ts)*1000
    arr_ts_std = np.nanstd(pred_arr_ts)*1000

    mean_errors_arr_len = np.nanmean(errors_arr_len)
    arr_len_std = np.nanstd(pred_arr_len)

    mean_errors_seg_ts = np.nanmean(errors_seg_ts, axis=0)*1000
    seg_ts_std = np.nanstd(pred_seg_ts, axis=0)*1000

    mean_errors_seg_len = np.nanmean(errors_seg_len, axis=0)
    seg_len_std = np.nanstd(pred_seg_len, axis=0)

    mean_errors_seg_mretx = np.nanmean(errors_seg_mretx, axis=0)
    seg_mretx_std = np.nanstd(pred_seg_mretx, axis=0)

    mean_errors_seg_rfailed = np.nanmean(errors_seg_rfailed, axis=0)
    seg_rfailed_std = np.nanstd(pred_seg_rfailed, axis=0)

    return mean_errors_delay, pred_delay_std, mean_errors_arr_ts, arr_ts_std, mean_errors_arr_len, arr_len_std, \
        mean_errors_seg_ts, seg_ts_std, mean_errors_seg_len, seg_len_std, \
        mean_errors_seg_mretx, seg_mretx_std, mean_errors_seg_rfailed, seg_rfailed_std

def fix_packets_label(entry, num_packets):
    # fix the label packets
    packets_label = []
    assert entry["scheduling"][-1]['segment'] == -1
    sequence = [ entry["scheduling"][-1] ]
    for event in entry["label"]:
        if event['segment'] == -1 and len(sequence) > 1:
            packets_label.append(copy.deepcopy(sequence))
            sequence = []
        if len(packets_label) >= num_packets:
            break
        sequence.append(event)
    return packets_label

def get_pred_results(pred_seq_arr):
    num_pred_samples = len(pred_seq_arr)
    pred_seg_mretx = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_rfailed = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_ts = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_len = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_seg_num_rbs = np.zeros((num_pred_samples, MAX_NUM_SEGMENTS))
    pred_arrival_ts = np.zeros((num_pred_samples))
    pred_arrival_len = np.zeros((num_pred_samples))
    pred_departure_ts = np.zeros((num_pred_samples))
    for idx, pred_seq in enumerate(pred_seq_arr): # idx iterates over the samples
        pred_arrival_ts[idx] = pred_seq[0]['timestamp']
        pred_arrival_len[idx] = pred_seq[0]['len']
        pred_departure_ts[idx] = pred_seq[-1]['timestamp'] + float(pred_seq[-1]['mretx'])*0.01 + 0.001 # 1ms for departure
        for idy in range(MAX_NUM_SEGMENTS):
            if len(pred_seq)-1 <= idy: # minus one for arrival
                pred_seg_mretx[idx,idy] = np.nan
                pred_seg_rfailed[idx,idy] = np.nan
                pred_seg_ts[idx,idy] = np.nan
                pred_seg_len[idx,idy] = np.nan
                pred_seg_num_rbs[idx,idy] = np.nan
            else:
                pred_seg_mretx[idx,idy] = pred_seq[idy+1]['mretx']
                pred_seg_rfailed[idx,idy] = pred_seq[idy+1]['rfailed']
                pred_seg_ts[idx,idy] = pred_seq[idy+1]['timestamp']
                pred_seg_len[idx,idy] = pred_seq[idy+1]['len']
                pred_seg_num_rbs[idx,idy] = pred_seq[idy+1]['num_rbs']

    return pred_arrival_ts, pred_arrival_len, pred_departure_ts, pred_seg_ts, pred_seg_len, pred_seg_num_rbs, pred_seg_mretx, pred_seg_rfailed

def get_label_results(label_seq):
    label_seg_mretx = np.zeros((MAX_NUM_SEGMENTS))
    label_seg_rfailed = np.zeros((MAX_NUM_SEGMENTS))
    label_seg_ts = np.zeros((MAX_NUM_SEGMENTS))
    label_seg_len = np.zeros((MAX_NUM_SEGMENTS))
    label_seg_num_rbs = np.zeros((MAX_NUM_SEGMENTS))
    label_arrival_ts = label_seq[0]['timestamp']
    label_arrival_len = label_seq[0]['len']
    label_departure_ts = label_seq[0]['depart_timestamp']
    for idy in range(MAX_NUM_SEGMENTS):
        if len(label_seq)-1 <= idy:
            label_seg_mretx[idy] = np.nan
            label_seg_rfailed[idy] = np.nan
            label_seg_ts[idy] = np.nan
            label_seg_len[idy] = np.nan
            label_seg_num_rbs[idy] = np.nan
        else:
            label_seg_mretx[idy] = label_seq[idy+1]['mretx']
            label_seg_rfailed[idy] = label_seq[idy+1]['rfailed']
            label_seg_ts[idy] = label_seq[idy+1]['timestamp']
            label_seg_len[idy] = label_seq[idy+1]['len']
            label_seg_num_rbs[idy] = label_seq[idy+1]['num_rbs']

    return label_arrival_ts, label_arrival_len, label_departure_ts, label_seg_ts, label_seg_len, label_seg_num_rbs, label_seg_mretx, label_seg_rfailed


def evaluate_model(args):

    # open the files and load the data
    log_folder = Path(args.source) / "e2e" / "prediction_results" / args.name / args.id
    conf_file = next(log_folder.glob("*.json"))
    with open(conf_file, 'r') as file:
        e2e_config = json.load(file)
    num_packets = e2e_config["num_future_packet_predictions"]

    with open(log_folder / "result.pkl", 'rb') as file:
        dataset = pickle.load(file)

    num_batches = e2e_config["num_batches"]
    batch_size = e2e_config["batch_size"]    
    num_entries = num_batches*batch_size

    total_errors_delay = np.zeros((num_packets))
    total_pred_delay_std = np.zeros((num_packets))

    total_errors_arr_ts = np.zeros((num_packets))
    total_arr_ts_std = np.zeros((num_packets))
    total_errors_arr_len = np.zeros((num_packets))
    total_arr_len_std = np.zeros((num_packets))

    total_errors_seg_ts = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    total_seg_ts_std = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    total_errors_seg_len = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    total_seg_len_std = np.zeros((num_packets, MAX_NUM_SEGMENTS))

    total_errors_seg_mretx = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    total_seg_mretx_std = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    total_errors_seg_rfailed = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    total_seg_rfailed_std = np.zeros((num_packets, MAX_NUM_SEGMENTS))

    count_errors = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    count_stds = np.zeros((num_packets, MAX_NUM_SEGMENTS))
    for idk in range(num_entries):
        entry = dataset[idk]
        # fix the label packets
        packets_label = fix_packets_label(entry, num_packets)
        for idx, packet_pred in enumerate(entry["pred"]): # idx: packet number
            # check all first events are arrivals in predictions
            first_types_are_arrival = ~(np.array([ packet_pred[i][0]['type_event'] for i in range(len(packet_pred)) ]).astype(bool))
            assert np.all(first_types_are_arrival), f"Assertion failed for packet {idx} in entry {idk}"
            if idx == 0:
                # first packet checks
                pred_arrival_ms = np.array([ item[0]['timestamp'] for item in packet_pred ])*1000
                assert np.allclose(pred_arrival_ms, pred_arrival_ms[0])
                assert abs(pred_arrival_ms[0] - (packets_label[idx][0]['timestamp']*1000)) < 0.5
            mean_errors_delay, pred_delay_std, mean_errors_arr_ts, \
            arr_ts_std, mean_errors_arr_len, arr_len_std, \
            mean_errors_seg_ts, seg_ts_std, mean_errors_seg_len, seg_len_std, \
            mean_errors_seg_mretx, seg_mretx_std, mean_errors_seg_rfailed, seg_rfailed_std = \
                calc_errors(packet_pred, packets_label[idx])

            total_errors_delay[idx] += mean_errors_delay
            total_pred_delay_std[idx] += pred_delay_std

            total_errors_arr_ts[idx] += mean_errors_arr_ts 
            total_arr_ts_std[idx] += arr_ts_std
            total_errors_arr_len[idx] += mean_errors_arr_len
            total_arr_len_std[idx] += arr_len_std

            # make a mask for nans
            mask = (~np.isnan(mean_errors_seg_ts)).astype(int)
            count_errors[idx,:] += mask
            mask = (~np.isnan(seg_ts_std)).astype(int)
            count_stds[idx,:] += mask

            total_errors_seg_ts[idx,:] += np.nan_to_num(mean_errors_seg_ts)
            total_seg_ts_std[idx,:] += np.nan_to_num(seg_ts_std)

            total_errors_seg_len[idx,:] += np.nan_to_num(mean_errors_seg_len)
            total_seg_len_std[idx,:] += np.nan_to_num(seg_len_std)

            total_errors_seg_mretx[idx,:] += np.nan_to_num(mean_errors_seg_mretx)
            total_seg_mretx_std[idx,:] += np.nan_to_num(seg_mretx_std)

            total_errors_seg_rfailed[idx,:] += np.nan_to_num(mean_errors_seg_rfailed)
            total_seg_rfailed_std[idx,:] += np.nan_to_num(seg_rfailed_std)

    
    total_errors_delay = total_errors_delay/num_entries
    total_pred_delay_std = total_pred_delay_std/num_entries

    print("Mean delay error: ", total_errors_delay)
    print("Mean delay std: ", total_pred_delay_std)

    total_errors_arr_ts = total_errors_arr_ts/num_entries
    total_arr_ts_std = total_arr_ts_std/num_entries
    total_errors_arr_len = total_errors_arr_len/num_entries
    total_arr_len_std = total_arr_len_std/num_entries

    print("Mean arrival time error: ", total_errors_arr_ts)
    print("Mean arrival time std: ", total_arr_ts_std)
    print("Mean arrival length error: ", total_errors_arr_len)
    print("Mean arrival length std: ", total_arr_len_std)

    total_errors_seg_ts = total_errors_seg_ts/count_errors
    total_seg_ts_std = total_seg_ts_std/count_stds
    total_errors_seg_len = total_errors_seg_len/count_errors
    total_seg_len_std = total_seg_len_std/count_stds

    print("Mean segment time error: ", total_errors_seg_ts)
    print("Mean segment time std: ", total_seg_ts_std)
    print("Mean segment length error: ", total_errors_seg_len)
    print("Mean segment length std: ", total_seg_len_std)

    total_errors_seg_mretx = total_errors_seg_mretx/count_errors
    total_seg_mretx_std = total_seg_mretx_std/count_stds

    print("Mean mretx error: ", total_errors_seg_mretx)
    print("Mean mretx std: ", total_seg_mretx_std)

    total_errors_seg_rfailed = total_errors_seg_rfailed/count_errors
    total_seg_rfailed_std = total_seg_rfailed_std/count_stds

    print("Mean rfailed error: ", total_errors_seg_rfailed)
    print("Mean rfailed std: ", total_seg_rfailed_std)

    
def plot_predictions(args):

    # open the files and load the data
    log_folder = Path(args.source) / "e2e" / "prediction_results" / args.name / args.id
    conf_file = next(log_folder.glob("*.json"))
    with open(conf_file, 'r') as file:
        e2e_config = json.load(file)
    num_packets = e2e_config["num_future_packet_predictions"]
    packet_num = 0
    assert packet_num < num_packets

    pkl_file = log_folder / "result.pkl"
    with open(pkl_file, 'rb') as file:
        dataset = pickle.load(file)

    num_batches = e2e_config["num_batches"]
    batch_size = e2e_config["batch_size"]    

    num_entries = num_batches*batch_size
    # pick a random number from 0 to num_entries
    idx = np.random.randint(num_entries)
    entry = dataset[idx]
    assert "pred" in entry

    packets_label = fix_packets_label(entry, num_packets)
    label_arrival_ts, label_arrival_len, label_departure_ts, \
        label_seg_ts, label_seg_len, label_seg_num_rbs, label_seg_mretx, label_seg_rfailed = \
            get_label_results(packets_label[packet_num])
    pred_arrival_ts, pred_arrival_len, pred_departure_ts, \
        pred_seg_ts, pred_seg_len, pred_seg_num_rbs, pred_seg_mretx, pred_seg_rfailed = \
            get_pred_results(entry["pred"][packet_num])
    num_pred_samples = pred_arrival_ts.shape[0]

    # Convert timestamps to milliseconds
    pred_arrival_ms = pred_arrival_ts * 1000  # Convert to ms
    # check if all pred_arrival_ms are the same
    assert np.allclose(pred_arrival_ms, pred_arrival_ms[0])
    # check if all pred_arrival_ts is equal to label_arrival_ts
    assert pred_arrival_ts[0] == label_arrival_ts

    pred_seg_ms = pred_seg_ts * 1000
    pred_departure_ms = pred_departure_ts * 1000
    label_seg_ms = label_seg_ts * 1000
    label_departure_ms = label_departure_ts * 1000

    # Set the origin of time to 2 ms before the earliest arrival
    time_origin = pred_arrival_ms[0]# packet 0 arrival time is the time origin
    adjusted_pred_seg_ms = pred_seg_ms - time_origin
    adjusted_pred_departure_ms = pred_departure_ms - time_origin
    adjusted_label_seg_ms = label_seg_ms - time_origin
    adjusted_label_departure_ms = label_departure_ms - time_origin

    # Set histogram parameters
    bin_width = 0.5  # in ms
    start = 0  # Origin is now 0
    end = 38  # 40 ms after the earliest arrival, accounting for the new origin
    bins = np.arange(start, end, bin_width)

    # Normalize the histogram to reflect probability
    weights = np.ones_like(adjusted_pred_departure_ms) / num_pred_samples
    # IEEE Transactions format settings
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,  # Font size as per IEEE standards
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "lines.linewidth": 1,
        "axes.linewidth": 0.8
    })

    # Create figure with 4 vertically stacked subplots
    fig, axs = plt.subplots(4, 2, figsize=(3, 4), sharex='col', gridspec_kw={'width_ratios': [0.9, 0.1], 'hspace': 0.2, 'wspace': 0.1})

    # Plot histograms on the left column
    for i in range(3):
        ax = axs[i, 0]
        ax.vlines(adjusted_label_seg_ms[i], ymin=0, ymax=1, color='red', linestyle='--', linewidth=1)
        ax.hist(adjusted_pred_seg_ms[:, i], bins=bins, weights=weights, color='orange', edgecolor='black', alpha=0.7, label=f"Prediction")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_ylim([0, 1])
        ax.text(0.95, 0.85, f"Segment {i+1}", transform=ax.transAxes, ha='right', va='top', fontsize=9)

        # Set custom x-axis ticks
        ax.set_xticks(range(0, int(end) + 1, 5))  # Set ticks every 5 ms
        ax.set_xlim([start, end])  # Ensure the x-axis limits match the histogram range

    # Plot the departure time histogram
    axs[3,0].vlines(adjusted_label_departure_ms, ymin=0, ymax=1, color='red', linestyle='--', linewidth=1)
    axs[3,0].hist(adjusted_pred_departure_ms, bins=bins, weights=weights, color='orange', edgecolor='black', alpha=0.7)
    axs[3,0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axs[3,0].set_ylim([0, 1])
    axs[3,0].set_xlabel("Time [ms]")
    axs[3, 0].text(0.95, 0.85, "Departure", transform=axs[3, 0].transAxes, ha='right', va='top', fontsize=9)

    # Plot box plots for 'len' on the right column (skip the bottom row)
    for i in range(3):
        ax = axs[i, 1]  # Primary axis for box plots

        # Calculate mean value for 'len'
        mean_len = np.mean(pred_seg_len[:, i])

        # Bar plot for mean value
        ax.bar(1, mean_len, width=0.5, color='blue', alpha=0.7, label='Mean Value')
        ax.axhline(label_seg_len[i], color='red', linestyle='--', linewidth=1, label="Ground Truth")

        ax.set_ylim([0, 250])  # Set limit for len
        ax.yaxis.set_label_position("right")  # Move y-axis to the right
        ax.yaxis.tick_right()  # Move ticks to the right
        #ax.set_ylabel("Bytes")

    # Add shared y-axis labels
    fig.text(0.005, 0.55, "Probability", va='center', rotation='vertical', fontsize=10)  # Left column
    fig.text(0.82, 0.27, "Bytes", va='center', rotation='horizontal', fontsize=10)  # Right column


    # Remove the bottom-right subplot
    fig.delaxes(axs[3, 1])

    # Shared X-axis label for the histograms
    axs[3, 0].set_xlabel("Time [ms]")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(log_folder / "time_histogram.pdf", format="pdf")

    # Save the figure as a pickle file
    with open(log_folder / "time_histogram.pkl", "wb") as f:
        pickle.dump(fig, f)

    # Close the figure to avoid showing it or consuming memory
    plt.close(fig)

    # Reset plt parameters to default
    plt.rcParams.update(plt.rcParamsDefault)


