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
base_path = "./data/s61-64_results/link_quality/prediction_results"

# Categories and history lengths to plot
categories = ['mcs_20k_h5_noisy', 'mcs_20k_h10_noisy', 'mcs_20k_h50_noisy', 'mcs_20k_h100_noisy']
versions = ['', '_head2']

# JSON keys to plot
keys_to_plot = [
    "loglike", "mcs_err", "mcs_change_err"
]

# Initialize dictionaries to store values for each key and version
values_by_key = {key: {version: {category: {} for category in categories} for version in versions} for key in keys_to_plot}

# Loop through each key, version, category, and history length
for key in keys_to_plot:
    for version in versions:
        for category in categories:
            folder_name = f"{category}{version}"
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
                                    values_by_key[key][version][category] = -value if key.endswith("loglike") else value
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
    "loglike": "Standardized Test NLL",
    "mcs_err": "MCS Index MAE",
    "mcs_change_err": "MCS Index Change MAE"
}

# Generate plots for each key
for key in keys_to_plot:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))  # IEEE standard figure size

    x = range(len(categories))

    # Plot lines for each version
    for version in versions:
        values = [values_by_key[key][version][category] for category in categories]
        label = "Encoder with 4 heads" if version == "" else "Encoder with 2 heads"
        ax.plot(x, values, marker='o', label=label)

    ax.set_ylabel(y_axis_labels[key])
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(['5', '10', '50', '100'])  # Adjust the x-axis labels
    ax.set_xlabel("History Length [events]")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the figure in PDF and pickle formats
    figure_filename = f"{key}_combined"
    plt.savefig(Path(base_path) / f"{figure_filename}.pdf", format='pdf')
    with open(Path(base_path) / f"{figure_filename}.pkl", 'wb') as f:
        pickle.dump(fig, f)

    plt.close(fig)
