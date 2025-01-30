from datetime import datetime
from plotly.subplots import make_subplots
import os
import json
import re
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import math
import numpy as np
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

# Ensure IEEE-compliant font and style
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

# Path to the `data` directory
dataset_path = Path("./data/s61-64_results/e2e/datasets/main_dataset")
databse_id = 1

# open the pickle file inside the main_dataset_path
with open(dataset_path / "dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# open the json file inside the main_dataset_path
with open(dataset_path / "config.json", "r") as f:
    dataset_config = json.load(f)

# Create the marginal delay distribution
# get all delay values of entries in the dataset

all_packet_arrivals = []
for idx, db in enumerate(dataset):
    packet_arrivals = []
    for idy in range(len(db['dataset'])):
        entry = db['dataset'][idy]
        if entry['segment'] == -1:
            packet_arrivals.append({
                "timestamp": entry["timestamp"],
                "depart_timestamp": entry["depart_timestamp"],
                "len": entry["len"], 'slot' : entry["slot"], 'mcs_index': entry["mcs_index"], 
                'mretx': entry["mretx"], 'rfailed':entry["rfailed"], 'num_rbs': entry["num_rbs"], 
                'num_symbols': entry["num_symbols"],
                "delay": (entry["depart_timestamp"]-entry["timestamp"])*1000,
            })
    for idy in range(len(packet_arrivals)-1):
        packet_arrivals[idy+1]["interarrival_time"] = (packet_arrivals[idy+1]["timestamp"] - packet_arrivals[idy]["timestamp"])*1000
    del packet_arrivals[0]
    all_packet_arrivals.append(packet_arrivals)
    delays = np.array([packet["delay"] for packet in packet_arrivals])
    interarrival_times = np.array([packet["interarrival_time"] for packet in packet_arrivals])
    packet_length = packet_arrivals[0]['len']
    logger.info(f"DB {idx}, packet length: {packet_length}, interarrival time: {interarrival_times.mean()} std:{interarrival_times.std()}, number of packets: {len(packet_arrivals)}, average delay: {delays.mean()}")

logger.info(f"Choosing databse {databse_id}")
packet_delay_pairs = all_packet_arrivals[databse_id]
marginal_delays = np.array([packet["delay"] for packet in packet_delay_pairs])
logger.info(f"Total number of packets: {len(packet_delay_pairs)}, average delay: {marginal_delays.mean()}")


# similar x for both marginal and conditional
x_lims = [10, 70]
y_lims = [1e-4, 1]
marginal_x = np.linspace(x_lims[0], x_lims[1], 1000).reshape(-1, 1)
num_bins = 500
kde_bw = 0.1


##########  PLOT  ##########
# Plot histogram of the marginal and conditional delays
fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))


##########  MARGINAL DISTRIBUTION OF DELAYS  ##########

# Fit a Kernel Density Estimator to the marginal delays
marginal_kde = KernelDensity(kernel='gaussian', bandwidth=kde_bw)
marginal_delays_reshaped = marginal_delays.reshape(-1, 1)
marginal_kde.fit(marginal_delays_reshaped)

# Generate a range of values for plotting the KDE
marginal_log_density = marginal_kde.score_samples(marginal_x)
marginal_pdf = np.exp(marginal_log_density)

# Plot the marginal histogram
marginal_hist, marginal_bins, _ = ax.hist(marginal_delays, bins=num_bins, density=True, alpha=0.6, label="All packets")

# Plot the KDE for marginal delays over the histogram
ax.plot(marginal_x, marginal_pdf, '--', color='C0')

##########


##########  CONDITIONAL DISTRIBUTION OF DELAYS - slot ##########
selected_slot_range = [15,16]
cond_packet_delay_pairs = [packet for packet in packet_delay_pairs if ((selected_slot_range[0] <= packet['slot']) and  (packet['slot'] <= selected_slot_range[1]))]
cond_delays = np.array([packet["delay"] for packet in cond_packet_delay_pairs])
logger.info(f"Total number of packets in slot range {selected_slot_range}: {len(cond_packet_delay_pairs)}, average delay: {cond_delays.mean()}")

# Fit a Kernel Density Estimator to the conditional delays
cond_kde = KernelDensity(kernel='gaussian', bandwidth=kde_bw)
cond_delays_reshaped = cond_delays.reshape(-1, 1)
cond_kde.fit(cond_delays_reshaped)

# Generate a range of values for plotting the KDE
cond_log_density = cond_kde.score_samples(marginal_x)
cond_pdf = np.exp(cond_log_density)

# Plot the conditional histogram using the bins from the marginal histogram
cond_hist, _, _ = ax.hist(cond_delays, bins=marginal_bins, density=True, alpha=0.6, label=f"Packets arrived on slot {selected_slot_range[0]}", color='C1')

# Plot the KDE for conditional delays over the histogram
ax.plot(marginal_x, cond_pdf, '--', color='C1')

##########

##########  CONDITIONAL DISTRIBUTION OF DELAYS - slot ##########
selected_slot_range = [1,2]
cond_packet_delay_pairs = [packet for packet in packet_delay_pairs if ((selected_slot_range[0] <= packet['slot']) and  (packet['slot'] <= selected_slot_range[1]))]
cond_delays = np.array([packet["delay"] for packet in cond_packet_delay_pairs])
logger.info(f"Total number of packets in slot range {selected_slot_range}: {len(cond_packet_delay_pairs)}, average delay: {cond_delays.mean()}")

# Fit a Kernel Density Estimator to the conditional delays
cond_kde = KernelDensity(kernel='gaussian', bandwidth=kde_bw)
cond_delays_reshaped = cond_delays.reshape(-1, 1)
cond_kde.fit(cond_delays_reshaped)

# Generate a range of values for plotting the KDE
cond_log_density = cond_kde.score_samples(marginal_x)
cond_pdf = np.exp(cond_log_density)

# Plot the conditional histogram using the bins from the marginal histogram
cond_hist, _, _ = ax.hist(cond_delays, bins=marginal_bins, density=True, alpha=0.6, label=f"Packets arrived on slot {selected_slot_range[0]}", color='C2')

# Plot the KDE for conditional delays over the histogram
ax.plot(marginal_x, cond_pdf, '--', color='C2')

##########


##########  CONDITIONAL DISTRIBUTION OF DELAYS - mcs ##########
selected_mcs_range = [14,15]
cond_packet_delay_pairs = [packet for packet in packet_delay_pairs if ((selected_mcs_range[0] <= packet['mcs_index']) and  (packet['mcs_index'] <= selected_mcs_range[1]))]
cond_delays = np.array([packet["delay"] for packet in cond_packet_delay_pairs])
logger.info(f"Total number of packets in mcs range {selected_mcs_range}: {len(cond_packet_delay_pairs)}, average delay: {cond_delays.mean()}")

# Fit a Kernel Density Estimator to the conditional delays
cond_kde = KernelDensity(kernel='gaussian', bandwidth=kde_bw)
cond_delays_reshaped = cond_delays.reshape(-1, 1)
cond_kde.fit(cond_delays_reshaped)

# Generate a range of values for plotting the KDE
cond_log_density = cond_kde.score_samples(marginal_x)
cond_pdf = np.exp(cond_log_density)

# Plot the conditional histogram using the bins from the marginal histogram
cond_hist, _, _ = ax.hist(cond_delays, bins=marginal_bins, density=True, alpha=0.6, label=f"Packets transmitted with mcs {selected_mcs_range[0]}", color='C3')

# Plot the KDE for conditional delays over the histogram
ax.plot(marginal_x, cond_pdf, '--', color='C3')

##########

ax.set_xlabel("Packet Delay [ms]")
ax.set_ylabel("Probability Density")
# Switch y to log scale
ax.set_yscale("log")
# Set y axis limits to [1e-5, 1]
ax.set_xlim(x_lims[0], x_lims[1])
ax.set_ylim(y_lims[0], y_lims[1])
ax.legend()
fig.tight_layout()
plt.savefig("combined_delay_distribution.pdf")
plt.close()


