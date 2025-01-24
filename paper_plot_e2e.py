from datetime import datetime
import os
import json
import re
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import math
import numpy as np

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

# Categories and history lengths to plot
categories = {
    'Type' : ['MLP', 'LSTM', 'Transformer'],
    'Forecasting Model' : ['Direct', 'Autoregressive'],
    'History Length': ['5', '20']
}

path_with_types = {
    'MLP': base_path / 'test10k_final_mlp',
    'LSTM': base_path / 'test10k_final_retx_rnn',
    'Transformer': base_path / 'test10k_final_retx_transformer'
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

# Helper to map model_id -> Type
model_type_map = {
    "MLPE2E": "MLP",
    "RecurrentE2E": "LSTM",
    "TransformerE2E": "Transformer"
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

def extract_training_log_details(file_path):
    num_parameters = None
    epoch_times = []
    train_start_pattern = re.compile(r"\[( Epoch \d+ \(train\) )\]: train loglike")
    epoch_time_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
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

    # Calculate average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else None
    return num_parameters, avg_epoch_time

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
data = {}


for model_type, model_path in path_with_types.items():
    if not model_path.exists():
        print(f"Path does not exist for model type {model_type}: {model_path}")
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
        
        # Extract model_id to figure out 'Type'
        model_id = yaml_config['model_config']['model_id']
        if model_id not in model_type_map:
            continue  # skip unknown model_id

        trained_model_path = Path(yaml_config['model_config']['pretrained_model_dir'])
        trained_model_path = trained_model_path.parent.parent # go back two levels
        training_log_file = trained_model_path / "log"
        num_params, avg_training_epoch_time = extract_training_log_details(training_log_file)
        
        
        actual_type = model_type_map[model_id]
        
        # For MLP, we do not have 'Forecasting Model' nor 'History Length'
        if actual_type == "MLP":
            # Key is (MLP, None, None)
            f_model = None
            h_len = None
            tgt_len = 1
        else:
            # LSTM / Transformer
            last_layer_mlp = yaml_config['model_config']['model_specs'].get('last_layer_mlp', False)
            f_model = "Direct" if last_layer_mlp else "Autoregressive"
            # Read history length if available (default 'Unknown' if not)
            h_len = str(yaml_config['model_config']['model_specs'].get('src_seq_len', 'Unknown'))
            tgt_len = int(yaml_config['model_config']['model_specs']['tgt_seq_len'])
        
        # Load JSON results
        with open(results_path, 'r') as f:
            results_json = json.load(f)
        
        # calc training and evaluation times per sequence
        num_sequences = int(results_json['num_events'])/tgt_len
        training_time = avg_training_epoch_time / num_sequences
        inference_time = evaluation_time / num_sequences

        # We only care about the fields in to_plot
        entry = {}
        entry['num_parameters'] = num_params
        entry['training_time'] = training_time
        entry['inference_time'] = inference_time
        entry['loglike']      = results_json.get('loglike', None)
        entry['mae']    = results_json.get('mae', None)
        entry['coverage_50']  = results_json.get('coverage_50', None)
        entry['coverage_70']  = results_json.get('coverage_70', None)
        entry['coverage_90']  = results_json.get('coverage_90', None)
        entry['coverage_99'] = results_json.get('coverage_99', None)
        entry['loglike_sw']      = results_json.get('loglike_sw', None)
        entry['mae_sw']    = results_json.get('mae_sw', None)
        entry['coverage_50_sw']  = results_json.get('coverage_50_sw', None)
        entry['coverage_70_sw']  = results_json.get('coverage_70_sw', None)
        entry['coverage_90_sw']  = results_json.get('coverage_90_sw', None)
        entry['coverage_99_sw'] = results_json.get('coverage_99_sw', None)
        
        data[(actual_type, f_model, h_len)] = entry

###############################################################################
# 2) Create plots
###############################################################################
# We create three figures:
#   1. loglike (Standardized Test NLL)
#   2. mae (Delay MAE)
#   3. coverage (70%, 90%, 99%, 99.9%)

# ---------------------------------------------------------------------------
# Helper function to build the x-ticks for "loglike" and "mae":
#  x-axis = different pairs of (type, forecasting_model) 
#  lines = different history_length
#  MLP has no forecasting_model nor history_length
# ---------------------------------------------------------------------------

def get_loglike_mae_plot_data(metric_key):
    """
    Returns:
        x_labels: list of unique x-axis labels (e.g., "MLP", "LSTM+Direct", "LSTM+Autoregressive", etc.)
        lines_data: dict of {history_length -> (y_values for each x_label)}
                    MLP will appear as a single point in each line.
    """
    # We want an ordered set of (type, forecast) for the x-axis
    # MLP is just ("MLP", None)
    # LSTM -> ("LSTM", "Direct") or ("LSTM", "Autoregressive")
    # Transformer -> ("Transformer", "Direct") or ("Transformer", "Autoregressive")
    x_axis_pairs = [
        ("MLP", None),
        ("LSTM", "Direct"),
        ("LSTM", "Autoregressive"),
        ("Transformer", "Direct"),
        ("Transformer", "Autoregressive")
    ]
    x_labels = []
    # lines_data is a dict: {history_length -> [list of metric_value in the order of x_axis_pairs]}
    lines_data = {}
    
    # Build the label for the x-axis
    for t, f in x_axis_pairs:
        if t == "MLP":
            x_labels.append("MLP")
        else:
            x_labels.append(f"{t} + {f}")
    
    # Initialize lines for each known history length
    for hist_len in categories['History Length']:
        lines_data[hist_len] = []
    
    # Fill data
    for (model_t, model_f) in x_axis_pairs:
        if model_t == "MLP":
            # Single entry for MLP -> (MLP, None, None)
            mlp_entry = data.get((model_t, None, None), {})
            val = mlp_entry.get(metric_key, float('nan'))
            # put that value in each lines_data for each hist
            for hist_len in categories['History Length']:
                lines_data[hist_len].append(val)
        else:
            # LSTM or Transformer, for each hist
            for hist_len in categories['History Length']:
                entry = data.get((model_t, model_f, hist_len), {})
                val = entry.get(metric_key, float('nan'))
                lines_data[hist_len].append(val)
    
    return x_labels, lines_data



# ---------------------------------------------------------------------------
# Plot 1: Log-likelihood
# ---------------------------------------------------------------------------
x_labels, lines_data = get_loglike_mae_plot_data("loglike")
fig, ax = plt.subplots(figsize=(6, 3))  # adjust figsize as needed
x_positions = range(len(x_labels))  # for the x-axis

markers = ['o', 's', '^', 'D']  # for the different lines
for i, hist_len in enumerate(categories['History Length']):
    ax.plot(
        x_positions, 
        -np.array(lines_data[hist_len]),
        marker=markers[i % len(markers)],
        label=f"History={hist_len}"
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=25, ha="right")
ax.set_ylabel(y_axis_labels["loglike"])
ax.set_title("Log-likelihood Comparison")
ax.legend()
fig.tight_layout()

# Save figure
loglike_fig_path = base_path / "loglike_plot.pdf"
fig.savefig(loglike_fig_path)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: MAE
# ---------------------------------------------------------------------------
x_labels, lines_data = get_loglike_mae_plot_data("mae")
fig, ax = plt.subplots(figsize=(6, 3))
x_positions = range(len(x_labels))

for i, hist_len in enumerate(categories['History Length']):
    ax.plot(
        x_positions,
        lines_data[hist_len],
        marker=markers[i % len(markers)],
        label=f"History={hist_len}"
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=25, ha="right")
ax.set_ylabel(y_axis_labels["mae"])
ax.set_title("Mean Absolute Error Comparison")
ax.legend()
fig.tight_layout()

# Save figure
mae_fig_path = base_path / "mae_plot.pdf"
fig.savefig(mae_fig_path)
plt.close(fig)

# ---------------------------------------------------------------------------
# Updated Coverage Plot: "Coverage vs. Desired Quantile"
# ---------------------------------------------------------------------------
# x‑axis: desired quantile (log(1-q))
# y‑axis: empirical coverage (log(1-q))
# A diagonal line represents perfect calibration.

# Build desired quantiles and corresponding coverage keys
desired_quantiles = [0.5, 0.7, 0.9, 0.99]  # Example desired quantiles
coverage_keys = ["coverage_50", "coverage_70", "coverage_90", "coverage_99"]

# Compute log(1-q) for desired quantiles
log_1_q_desired = np.log10(1 - np.array(desired_quantiles))

# Prepare data for empirical coverage and log(1-q)
empirical_coverage = {}  # Store log(1-q) values for empirical coverage

for key, val_dict in data.items():
    model_t, f_model, h_len = key
    
    # Build legend label
    if model_t == "MLP":
        label = "MLP"
    else:
        label = f"{model_t} + {f_model} + H={h_len}"
    
    # Extract empirical coverage for this model
    coverages = [val_dict.get(ck, float('nan')) for ck in coverage_keys]
    log_1_q_empirical = np.log10(1 - np.array(coverages))
    empirical_coverage[label] = log_1_q_empirical

# Plot Coverage vs. Desired Quantile
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the diagonal line for perfect calibration
ax.plot(
    log_1_q_desired,
    log_1_q_desired,
    linestyle="--",
    color="black",
    label="Perfect Calibration"
)

# Plot each model's empirical coverage
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'x']  # Different markers for models
for i, (label, log_1_q_empirical) in enumerate(empirical_coverage.items()):
    ax.plot(
        log_1_q_desired,
        log_1_q_empirical,
        marker=markers[i % len(markers)],
        label=label
    )

# Configure the plot
ax.set_xlabel("Desired Coverage [log(1-q)]")
ax.set_ylabel("Empirical Coverage [log(1-q)]")
ax.legend(fontsize=8)
ax.grid(True)
fig.tight_layout()

# Save the updated figure
coverage_plot_path = base_path / "coverage.pdf"
fig.savefig(coverage_plot_path)
plt.close(fig)

print(f"Updated coverage plot saved to: {coverage_plot_path}")

print("Plots saved to:")
print(f" - {loglike_fig_path}")
print(f" - {mae_fig_path}")
print(f" - {coverage_plot_path}")



###############################################################################
# 3) Create Step‑Wise Figures
###############################################################################
# We will add 3 more plots, each “step-wise”:
#   (A) mae vs. step
#   (B) var vs. step
#   (C) coverage vs. step (with subplots for coverage_70_sw, coverage_90_sw, etc.)
#
# Each model appears as a line in the same figure. The x-axis is the future step.
###############################################################################

# A small helper to build a label for each (type,f_model,h_len)
def build_model_label(model_t, f_model, h_len):
    """Returns a readable label like 'MLP', or 'LSTM+Direct+H=5'."""
    if model_t == "MLP":
        return "MLP"
    else:
        return f"{model_t}+{f_model}+H={h_len}"


# (0) Step‑wise loglike
fig, ax = plt.subplots(figsize=(6, 3))
for (model_t, f_model, h_len), val_dict in data.items():
    if val_dict['loglike_sw'] is None:
        continue  # skip if not available
    steps = range(len(val_dict['loglike_sw']))
    ax.plot(
        steps,
        val_dict['loglike_sw'],
        label=build_model_label(model_t, f_model, h_len)
    )

ax.set_xlabel("Future Step")
ax.set_ylabel("Step-wise Standardized NLL")
ax.set_title("Step-wise NLL Across Future Steps")
ax.legend(fontsize=8)
ax.grid(True)
fig.tight_layout()

mae_sw_fig_path = base_path / "loglike_stepwise.pdf"
fig.savefig(mae_sw_fig_path)
plt.close(fig)

# (A) Step‑wise mae
fig, ax = plt.subplots(figsize=(6, 3))
for (model_t, f_model, h_len), val_dict in data.items():
    if val_dict['mae_sw'] is None:
        continue  # skip if not available
    steps = range(len(val_dict['mae_sw']))
    ax.plot(
        steps,
        val_dict['mae_sw'],
        label=build_model_label(model_t, f_model, h_len)
    )

ax.set_xlabel("Future Step")
ax.set_ylabel("Step-wise MAE [ms]")
ax.set_title("Step-wise MAE Across Future Steps")
ax.legend(fontsize=8)
ax.grid(True)
fig.tight_layout()

mae_sw_fig_path = base_path / "mae_stepwise.pdf"
fig.savefig(mae_sw_fig_path)
plt.close(fig)


# (C) Step‑wise coverage (in log10(1 - coverage)): We'll create one figure with 4 subplots
# We also add a horizontal line for the perfect coverage at each coverage key.
coverage_sw_keys = [
    ("coverage_50_sw", 0.50,  "50%"),
    ("coverage_70_sw", 0.70,  "70%"),
    ("coverage_90_sw", 0.90,  "90%"),
    ("coverage_99_sw", 0.99,  "99%")
]

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
axes = axes.flatten()

def build_model_label(model_t, f_model, h_len):
    """Returns a readable label like 'MLP', or 'LSTM+Direct+H=5'."""
    if model_t == "MLP":
        return "MLP"
    else:
        return f"{model_t}+{f_model}+H={h_len}"

for ax, (cov_key, nominal_q, cov_label) in zip(axes, coverage_sw_keys):
    # Perfect coverage line:
    # For coverage = nominal_q, the log(1 - coverage) is log10(1 - nominal_q).
    perfect_coverage_log = np.log10(1.0 - nominal_q)
    
    # Plot a horizontal line for "perfect" coverage
    # We'll do it across the entire range of steps, so we'll just do a small trick:
    ax.axhline(perfect_coverage_log, color='black', linestyle='--', 
               label=f"Perfect {cov_label} (log10(1-{nominal_q}))")

    # Plot each model's stepwise coverage
    for (model_t, f_model, h_len), val_dict in data.items():
        sw_vals = val_dict.get(cov_key, None)
        if sw_vals is None:
            continue
        
        # Convert coverage to log10(1 - coverage)
        coverage_array = np.array(sw_vals)
        coverage_log = np.log10(1.0 - coverage_array)
        steps = range(len(coverage_array))
        
        ax.plot(
            steps,
            coverage_log,
            label=build_model_label(model_t, f_model, h_len)
        )
        
    ax.set_title(f"Coverage {cov_label}")
    ax.grid(True)
    ax.set_ylabel("log10(1 - coverage)")

# Shared X-label on bottom subplots
axes[-1].set_xlabel("Future Step")
axes[-2].set_xlabel("Future Step")

# Because we have multiple lines, let’s place a single legend in the last subplot or outside:
axes[-1].legend(fontsize=8, loc="best")

fig.suptitle("Step-wise Coverage in log10(1 - coverage)", fontsize=10)
fig.tight_layout()

coverage_sw_fig_path = base_path / "coverage_stepwise_log10.pdf"
fig.savefig(coverage_sw_fig_path)
plt.close(fig)

print(f"Step-wise coverage with log10(1-q) plot saved to {coverage_sw_fig_path}")



# ---------------------------------------------------------------------------
# Plot: Number of Parameters
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3))
x_labels = []
y_values = []

# Extract data for the plot
for (model_t, f_model, h_len), val_dict in data.items():
    if (model_t, f_model) not in x_labels:  # Only add once per model type
        x_labels.append((model_t, f_model))
        y_values.append(val_dict.get("num_parameters", 0))

x_labels = [f"{t} + {f}" if f else t for t, f in x_labels]
ax.bar(x_labels, y_values, color="skyblue")
ax.set_xlabel("Model Type")
ax.set_ylabel("Number of Parameters")
ax.set_title("Number of Parameters per Model")
ax.grid(axis="y")
fig.tight_layout()

# Save the figure
params_plot_path = base_path / "parameters_plot.pdf"
fig.savefig(params_plot_path)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: Training Time per Sequence
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3))
x_labels = []
y_values = []

# Extract data for the plot
for (model_t, f_model, h_len), val_dict in data.items():
    if (model_t, f_model) not in x_labels:  # Only add once per model type
        x_labels.append((model_t, f_model))
        y_values.append(val_dict.get("training_time", 0))

x_labels = [f"{t} + {f}" if f else t for t, f in x_labels]
ax.bar(x_labels, y_values, color="lightgreen")
ax.set_xlabel("Model Type")
ax.set_ylabel("Training Time per Sequence (seconds)")
ax.set_title("Training Time per Sequence")
ax.grid(axis="y")
fig.tight_layout()

# Save the figure
training_time_plot_path = base_path / "training_time_plot.pdf"
fig.savefig(training_time_plot_path)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: Inference Time per Sequence
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3))
x_labels = []
y_values = []

# Extract data for the plot
for (model_t, f_model, h_len), val_dict in data.items():
    if (model_t, f_model) not in x_labels:  # Only add once per model type
        x_labels.append((model_t, f_model))
        y_values.append(val_dict.get("inference_time", 0))

x_labels = [f"{t} + {f}" if f else t for t, f in x_labels]
ax.bar(x_labels, y_values, color="coral")
ax.set_xlabel("Model Type")
ax.set_ylabel("Inference Time per Sequence (seconds)")
ax.set_title("Inference Time per Sequence")
ax.grid(axis="y")
fig.tight_layout()

# Save the figure
inference_time_plot_path = base_path / "inference_time_plot.pdf"
fig.savefig(inference_time_plot_path)
plt.close(fig)

# ---------------------------------------------------------------------------
# Print Saved Paths
# ---------------------------------------------------------------------------
print(f"Plots saved to:")
print(f" - {params_plot_path}")
print(f" - {training_time_plot_path}")
print(f" - {inference_time_plot_path}")