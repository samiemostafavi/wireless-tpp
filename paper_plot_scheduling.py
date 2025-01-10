import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import math

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
base_path = "./data/multi_size_scheduling/scheduling/prediction_results"

# Categories and history lengths to plot
categories = ['test5k', 'test10k', 'test20k']
history_lengths = ['h5', 'h10', 'h20']
versions = ['', '-m']  # Plain and '-m' versions

# JSON keys to plot
keys_to_plot = [
    "loglike", "dtime_loglike", "len_loglike", 
    "dtime_mae", "len_mae", "dtime_var", "len_var"
]

# Initialize dictionaries to store values for each key and version
values_by_key = {key: {version: {category: {} for category in categories} for version in versions} for key in keys_to_plot}

# Loop through each key, version, category, and history length
for key in keys_to_plot:
    for version in versions:
        for category in categories:
            for history_length in history_lengths:
                folder_name = f"{category}-{history_length}{version}"
                category_path = os.path.join(base_path, folder_name)
                
                if os.path.exists(category_path):
                    # Find the subdirectory with a random name
                    subdirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
                    if subdirs:
                        random_dir = os.path.join(category_path, subdirs[0])  # Assuming only one random subdir exists
                        
                        # Find the JSON file in the random directory
                        json_files = [f for f in os.listdir(random_dir) if f.endswith('.json')]
                        if json_files:
                            json_file_path = os.path.join(random_dir, json_files[0])  # Assuming only one JSON file exists
                            
                            # Load the JSON file
                            try:
                                with open(json_file_path, 'r') as file:
                                    data = json.load(file)
                                    value = data.get(key, None)
                                    if value is not None:
                                        if key in ["dtime_var", "len_var"]:
                                            value = math.sqrt(value)  # Convert variance to standard deviation
                                        values_by_key[key][version][category][history_length] = -value if key.endswith("loglike") else value
                            except Exception as e:
                                print(f"Error reading JSON file in {random_dir}: {e}")
                        else:
                            print(f"No JSON file found in {random_dir}")
                    else:
                        print(f"No subdirectory found in {category_path}")
                else:
                    print(f"Category path does not exist: {category_path}")

# Y-axis labels
y_axis_labels = {
    "loglike": "NLL",
    "dtime_loglike": "Interarrival time NLL",
    "len_loglike": "Size NLL",
    "dtime_mae": "Interarrival time MAE [ms]",
    "len_mae": "Size MAE [Bytes]",
    "dtime_var": "Interarrival time STD [ms]",
    "len_var": "Size STD [Bytes]"
}

# Generate plots for each key
for key in keys_to_plot:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))  # IEEE standard figure size

    x = range(len(categories))
    width = 0.2  # Adjusted width for better visibility

    # Plot plain version bars for h5, h10, h20
    for i, history_length in enumerate(history_lengths):
        values = [values_by_key[key][''][category].get(history_length, 0) for category in categories]
        ax.bar([p + i * width for p in x], values, width=width, label=f"{history_length.split('h')[-1]} (base)")

    # Add h20 from -m
    values_h20_m = [values_by_key[key]['-m'][category].get('h20', 0) for category in categories]
    ax.bar([p + 3 * width for p in x], values_h20_m, width=width, label="20 (extended)", color='gray')

    ax.set_ylabel(y_axis_labels[key])
    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_xticklabels(['5k', '10k', '20k'])
    ax.legend(title="History Length")
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the figure in PDF and pickle formats
    figure_filename = f"{key}_combined"
    plt.savefig(Path(base_path) / f"{figure_filename}.pdf", format='pdf')
    with open(Path(base_path) / f"{figure_filename}.pkl", 'wb') as f:
        pickle.dump(fig, f)

    plt.close(fig)
