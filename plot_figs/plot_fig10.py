import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Define mappings for datasets and methods
dataset_name_mapping = {
    "census": "Census Income",
    "ufrgs": "UFRGS",
    "compas": "COMPAS",
    "diabetes": "Diabetes",
    "default": "Default Credit"
}

method_name_mapping = {
    "default": "default",
    "rew": "REW",
    "dir": "DIR",
    "fairsmote": "Fair-Smote",
    "meta": "META*",
    "adv": "ADV*",
    "pr": "PR*",
    "eop": "EOP",
    "ceo": "CEO",
    "roc": "ROC",
    "maat": "MAAT",
    "fairmask": "FairMask",
    "hifi_cpu": "HIFI* (CPU)",
    "hifi_gpu": "HIFI* (GPU)"
}

# Read the data file
file_path = osp.join(osp.dirname(__file__), "../results", "time_cost.txt")

# Manually parse the data
with open(file_path, 'r') as file:
    lines = file.readlines()

# Separate data into LR and DL sections
lr_lines = []
dl_lines = []
current_section = None
for line in lines:
    if line.strip() == 'lr:':
        current_section = 'lr'
    elif line.strip() == 'dl:':
        current_section = 'dl'
    elif current_section == 'lr':
        lr_lines.append(line)
    elif current_section == 'dl':
        dl_lines.append(line)

# Process the LR data, setting timeout to NaN for easier handling, and removing 'default' method
lr_df = pd.DataFrame([line.split() for line in lr_lines[1:] if line.strip()], columns=['method', 'census', 'ufrgs', 'compas', 'diabetes', 'default'])
lr_df.set_index('method', inplace=True)
lr_df.replace("timeout", np.nan, inplace=True)  # Replace 'timeout' with NaN for plotting
lr_df = lr_df.apply(pd.to_numeric, errors='coerce')
lr_df.drop(index='default', inplace=True, errors='ignore')  # Remove 'default' method if present

# Process the DL data, setting timeout to NaN, and removing 'default' method
dl_df = pd.DataFrame([line.split() for line in dl_lines[1:] if line.strip()], columns=['method', 'census', 'ufrgs', 'compas', 'diabetes', 'default'])
dl_df.set_index('method', inplace=True)
dl_df.replace("timeout", np.nan, inplace=True)
dl_df = dl_df.apply(pd.to_numeric, errors='coerce')
dl_df.drop(index='default', inplace=True, errors='ignore')  # Remove 'default' method if present

# Define specific colors, styles, and markers for each method to ensure consistency between plots
method_styles = {
    'rew': ('orange', '-', 's'),           # solid line with square marker
    'dir': ('orange', '--', 'D'),          # dashed line with diamond marker
    'fairsmote': ('orange', '-.', '^'),    # dash-dot line with triangle marker
    'meta': ('green', '-', 'o'),           # solid line with circle marker
    'adv': ('green', '--', 's'),           # dashed line with square marker
    'pr': ('green', '-.', 'D'),            # dash-dot line with diamond marker
    'eop': ('red', '-', 'o'),              # solid line with circle marker
    'ceo': ('red', '--', 's'),             # dashed line with square marker
    'roc': ('red', ':', 'D'),              # dotted line with diamond marker
    'maat': ('purple', '-', 'o'),          # solid line with circle marker
    'fairmask': ('purple', '--', 's'),     # dashed line with square marker
    'hifi_cpu': ('cyan', '-', 'o'),        # solid line with circle marker
    'hifi_gpu': ('cyan', '--', 's')        # dashed line with square marker
}

# Create a figure with two vertically stacked subplots
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Function to plot each method on a given axis with timeout handling
def plot_on_axis(df, ax, ylabel):
    for method, (color, linestyle, marker) in method_styles.items():
        if method in df.index:
            # Plot line for non-timeout data points
            ax.plot(df.columns, df.loc[method], marker=marker, label=method_name_mapping.get(method, method),
                    color=color, linestyle=linestyle, markersize=6)
            # Mark timeouts with 'X'
            timeout_indices = df.loc[method].isna()
            if timeout_indices.any():
                ax.scatter(df.columns[timeout_indices], [0] * sum(timeout_indices), marker='X', color=color,
                           label=f"{method_name_mapping.get(method, method)} (timeout)")
    ax.set_ylabel(ylabel)  # Set y-axis label for each subplot
    ax.set_xticks(range(len(df.columns)))  # Set x-ticks for both subplots
    ax.set_xticklabels([dataset_name_mapping.get(col, col) for col in df.columns])  # Use full names for datasets
    ax.grid()

# Plot LR and DL on respective subplots with updated y-axis labels
plot_on_axis(lr_df, ax1, "LR Model: Time Cost (seconds)")
plot_on_axis(dl_df, ax2, "NN Model: Time Cost (seconds)")

# Create a single legend outside the plot
handles, labels = ax1.get_legend_handles_labels()  # Get legend from the first axis (they are identical)
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), title="Fairness Methods", borderaxespad=0)

# Adjust layout to make sure everything fits in the figure
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Reserve space on the right for the legend
plt.savefig("combined_time_cost.png", dpi=300, bbox_inches='tight')  # Save as a single image with tight bounding box
plt.show()