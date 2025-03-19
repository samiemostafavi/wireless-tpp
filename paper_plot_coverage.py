from datetime import datetime
import os, copy
import json
import re
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import math
import numpy as np
import re

# Define markers for different model types
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']

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
base_path = Path("./data/intervals_results/e2e/prediction_results")
#base_path = Path("./data/s61-64_results/e2e/prediction_results")

figures_path = Path("./figures_intervals")
#figures_path = Path("./figures_intervals_final")
#figures_path = Path("./figures_s61-64_NEW")
#figures_path = Path("./figures_s61-64_EXC")

# Create the figures directory if it doesn't exist
figures_path.mkdir(parents=True, exist_ok=True)

# Categories and history lengths to plot
categories = {
    'Training Size': [1, 2.5, 5, 10],
    'Type' : ['mlp', 'lstm', 'lstmsingle', 'transformer'], # 'mlp', 'lstm', 'transformer', 'lstmmlp', 'transformermlp'
    'Window Length': [5, 20, 50, 100],
    'Auxiliary': ["NEW"] # 't8', 't12', 't20', 't24', 'noretx'
}

type_labels = {
    'mlp': 'MLP',
    'lstmsingle': 'LSTM-SS',
    'lstm': 'LSTM',
    'transformer': 'Transformer'
}

# Plot these
to_plot = {
    "loglike" : "Negative Log-Likelihood", 
    "mae" : "Mean Absolute Error [ms]", 
    "coverage": {
        "coverage_50" : "50% Coverage",
        "coverage_70" : "70% Coverage",
        "coverage_90" : "90% Coverage",
        "coverage_99" : "99% Coverage"
    }
}

# Y-axis labels
y_axis_labels = {
    "loglike": "Standardized Test NLL",
    "mae": "Delay MAE [ms]",
    "coverage": "Coverage Error [log]"
}


# check all folders inside the path_with_types, open the yaml file in them and get these:
# yaml_config['model_config']['model_specs']['last_layer_mlp'] corresponds to 'Forecasting Model'
#   if last_layer_mlp is True, it is 'Direct', otherwise 'Autoregressive'
# yaml_config['model_config']['model_id'] corresponds to 'Type'
#   if model_id is 'MLPE2E', it is 'MLP', if it is 'RecurrentE2E', it is 'LSTM', if it is 'TransformerE2E', it is 'Transformer'.
# once figured out the category of the model, remember its address. NOTE: MLP model is only one kind, it does not have 'Forecasting Model' nor 'History Length'.
# then, open the json file in the address, get the values for the keys in to_plot, and plot them. 
# we create three figures:
# 1. loglike
# 2. mae
# 3. coverage
# an example of the json file:
# {
#    "loglike": -0.7128046555955718,
#    "num_events": 587980.0,
#    "mae": 1.0770172199573138,
#    "var": 5.600416660308838,
#    "coverage_70": 0.6361661961291201,
#    "coverage_90": 0.9181502772203136,
#    "coverage_99": 0.9683832783428008,
#    "coverage_999": 0.9843464063403518
#}
# for the loglike and mae plots, let's have on x axis different pairs of (type, forecasting model) and then let's put the history length as a different line on the same plot to show it via a different marker and legend. Note that for MLP does not have the 'Forecasting Model' nor 'History Length' categories.
# for the coverage plot, let's have on x axis the quantile (70%, 90%, 99%, 999%), and different tuples of (type, forecasting model, and history length) on different lines on the same plot to show them via a different marker and legend.
# don't show the figures, just create them and save them in base_path / f"{figure_filename}.pdf".


###############################################################################
# HELPER FUNCTIONS
###############################################################################

def float_to_str(value):
    # Convert value to a string if it's not already
    value_str = str(float(value))
    
    # Remove leading zeros
    value_str = re.sub(r'^0+(\d)', r'\1', value_str)
    
    # Split into integer and decimal parts
    if '.' in value_str:
        integer_part, decimal_part = value_str.split('.')
        decimal_part = decimal_part.rstrip('0')  # Remove trailing zeros
        if decimal_part:
            return f"{integer_part}p{decimal_part}"
        else:
            return integer_part  # If decimal part is empty after stripping, return only integer
    else:
        return value_str  # Return as-is if no decimal point

def extract_training_log_details(file_path):
    num_parameters = None
    num_events = None
    epoch_times = []
    train_start_pattern = re.compile(r"\[( Epoch \d+ \(train\) )\]: train loglike")
    epoch_time_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
    num_events_pattern = re.compile(r"num_events is (\d+)")
    params_pattern = re.compile(r"Num of model parameters (\d+)")
    

    with open(file_path, 'r') as file:
        lines = file.readlines()
        prev_epoch_time = None

        for line in lines:
            # Extract number of parameters
            if num_parameters is None:
                params_match = params_pattern.search(line)
                if params_match:
                    num_parameters = int(params_match.group(1))

            # Extract epoch start and calculate time difference
            if train_start_pattern.search(line):
                time_match = epoch_time_pattern.search(line)
                if time_match:
                    current_epoch_time = time_match.group(1)
                    if prev_epoch_time:
                        time_diff = (
                            datetime.strptime(current_epoch_time, "%Y-%m-%d %H:%M:%S") -
                            datetime.strptime(prev_epoch_time, "%Y-%m-%d %H:%M:%S")
                        ).total_seconds()
                        epoch_times.append(time_diff)
                    prev_epoch_time = current_epoch_time

            if num_events is None:
                if num_events_pattern.search(line) and ("train loglike" in line):
                    num_events = int(num_events_pattern.search(line).group(1))

    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else None
    return num_parameters, avg_epoch_time, num_events

def extract_evaluation_log_details(file_path):
    evaluation_time = None
    evaluation_pattern = re.compile(r"Cost time: ([\d.]+)m")

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Extract evaluation cost time and convert to seconds
            eval_match = evaluation_pattern.search(line)
            if eval_match:
                evaluation_time = float(eval_match.group(1)) * 60  # Convert minutes to seconds
    return evaluation_time


###############################################################################
# 1) Collect data from each subfolder
###############################################################################

# This dictionary will store results in the form:
# data[(model_type, forecasting_model, history_length)] = {
#     'loglike': float,
#     'mae': float,
#     'coverage_70': float,
#     'coverage_90': float,
#     'coverage_99': float,
#     'coverage_999': float
# }

data = []
for s in categories['Training Size']:
    res_s = {
        "Training Size" : s,
        "Type" : None,
        "Window Length" : None,
        "Auxiliary" : None,
        "Path" : str(base_path / f"{float_to_str(s)}k")
    }
    for t in categories['Type']:
        res_t = copy.deepcopy(res_s)
        res_t["Type"] = t
        res_t["Path"] = res_s["Path"] + f"_{t}"
        for h in categories['Window Length']:
            res_h = copy.deepcopy(res_t)
            res_h["Window Length"] = h
            res_h["Path"] = res_t["Path"] + f"_{h}"
            for k in categories['Auxiliary']:
                res_k = copy.deepcopy(res_h)
                res_k["Auxiliary"] = k
                res_k["Path"] = res_h["Path"] + f"_{k}"
                data.append(res_k)
            if len(categories['Auxiliary']) == 0:
                data.append(res_h)


for model_dict in data:
    model_path = Path(model_dict['Path'])
    model_type = model_dict['Type']
    if not model_path.exists():
        print(f"Path does not exist for model path: {model_path}")
        continue
    
    # Each subfolder is expected to contain exactly one YAML file and exactly one JSON file
    for subfolder in model_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Find the YAML config
        yaml_files = list(subfolder.glob("*.yaml"))
        if len(yaml_files) != 1:
            # Either no YAML or more than one found, skip
            continue
        config_path = yaml_files[0]

        evaluation_log_file = subfolder / "log"
        evaluation_time = extract_evaluation_log_details(evaluation_log_file)
        
        # Find the JSON results
        json_files = list(subfolder.glob("*.json"))
        if len(json_files) != 1:
            # Either no JSON or more than one found, skip
            continue
        results_path = json_files[0]
        
        # Load YAML config
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)


        trained_model_path = Path(yaml_config['model_config']['pretrained_model_dir'])
        trained_model_path = trained_model_path.parent.parent # go back two levels
        training_log_file = trained_model_path / "log"
        num_params, avg_training_epoch_time, num_training_events = extract_training_log_details(training_log_file)
        
        # For MLP, we do not have 'Forecasting Model' nor 'History Length'
        if model_type == "mlp":
            h_len = 1 # this is training sequence length
            tgt_len = 1 # this is training sequence length
        else:
            # Read history length if available (default 'Unknown' if not)
            h_len = int(yaml_config['model_config']['model_specs'].get('src_seq_len', 'Unknown'))
            tgt_len = int(yaml_config['model_config']['model_specs']['tgt_seq_len'])
        
        # Load JSON results
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        
        # calc training and evaluation times per sequence
        
        num_sequences = num_training_events/h_len
        training_time = avg_training_epoch_time / num_sequences
        inference_time = evaluation_time / num_sequences

        # We only care about the fields in to_plot
        model_dict['num_parameters'] = num_params
        model_dict['training_time'] = training_time
        model_dict['inference_time'] = inference_time
        model_dict['results'] = results_json

# ---------------------------------------------------------------------------
# Updated Coverage Plot: "Coverage vs. Desired Quantile"
# ---------------------------------------------------------------------------
# x‑axis: desired quantile (log(1-q))
# y‑axis: empirical coverage (log(1-q))
# A diagonal line represents perfect calibration.

# For each window length, create and save a separate figure

desired_quantiles = [0.5, 0.7, 0.9, 0.99]  # Example desired quantiles
coverage_keys = ["coverage_50", "coverage_70", "coverage_90", "coverage_99"]

# Compute log(1-q) for desired quantiles
log_1_q_desired = np.log10(1 - np.array(desired_quantiles))

for train_size in categories['Training Size']:
    for w_length in categories['Window Length']:
        # Create a new figure
        plt.figure(figsize=(4, 2.5))

        # Plot a line for each model type
        for idx, model_type in enumerate(categories['Type']):
            # Filter data for the current type & window length
            subset = [
                r for r in data
                if r['Type'] == model_type
                and r['Window Length'] == w_length
                and r['Training Size'] == train_size
            ]
            assert len(subset) == 1
            # Extract x (Training Size) and y (loglike)
            x_vals = log_1_q_desired
            y_vals = [
                subset[0]['results']['coverage_50'], 
                subset[0]['results']['coverage_70'], 
                subset[0]['results']['coverage_90'],
                subset[0]['results']['coverage_99']
            ]
            y_vals = np.log10(1 - np.array(y_vals))

            # Plot if we have valid data
            plt.plot(x_vals, y_vals, marker=markers[idx % len(markers)], label=type_labels[model_type])

        # Plot the diagonal line for perfect calibration
        plt.plot(
            log_1_q_desired,
            log_1_q_desired,
            linestyle="--",
            color="black",
            label="Perfect Calibration"
        )

        # Labels, title, legend
        plt.xlabel("Target Coverage")
        plt.ylabel("Empirical Coverage")
        plt.legend(loc='lower right')
        plt.grid(True)

        # Set x and y ticks to the corresponding quantile values
        plt.xticks(log_1_q_desired, desired_quantiles)
        plt.yticks(log_1_q_desired, desired_quantiles)

        # Adjust layout and save to PDF
        plt.tight_layout()
        plt.savefig(figures_path / f"coverage_plot_s{train_size}_w{w_length}.pdf")
        plt.close()
