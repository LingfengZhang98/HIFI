"""
We use experimental results on NN models here.
"""

import matplotlib.pyplot as plt
import numpy as np


# (AOD, accuracy) tuples for HIFI with eta in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100, 1000]
coordinate_HIFI = [
    (0.197, 0.748),
    (0.187, 0.749),
    (0.177, 0.743),
    (0.153, 0.744),
    (0.170, 0.752),
    (0.157, 0.751),
    (0.147, 0.740),
    (0.126, 0.734),
    (0.118, 0.723),
    (0.095, 0.701),
    (0.055, 0.682),
    (0.048, 0.670)
]

# (AOD, accuracy) tuples for REW+HIFI with eta in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100, 1000]
coordinate_REW_HIFI = [
    (0.122, 0.740),
    (0.111, 0.740),
    (0.121, 0.741),
    (0.107, 0.738),
    (0.109, 0.739),
    (0.103, 0.738),
    (0.102, 0.732),
    (0.086, 0.729),
    (0.077, 0.719),
    (0.066, 0.703),
    (0.033, 0.656),
    (0.025, 0.662)
]

# (AOD, accuracy) tuples for MAAT+HIFI with eta in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100, 1000]
coordinate_MAAT_HIFI = [
    (0.144, 0.752),
    (0.1441, 0.752),
    (0.1381, 0.752),
    (0.138, 0.753),
    (0.139, 0.753),
    (0.1361, 0.750),
    (0.136, 0.753),
    (0.134, 0.752),
    (0.129, 0.751),
    (0.1221, 0.746),
    (0.123, 0.745),
    (0.122, 0.742)
]

# (AOD, accuracy) tuples for FairMask+HIFI with eta in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100, 1000]
coordinate_FairMask_HIFI = [
    (0.102, 0.731),
    (0.096, 0.732),
    (0.095, 0.739),
    (0.091, 0.740),
    (0.096, 0.747),
    (0.094, 0.745),
    (0.089, 0.734),
    (0.081, 0.731),
    (0.0811, 0.722),
    (0.072, 0.703),
    (0.043, 0.681),
    (0.031, 0.669)
]

# (AOD, accuracy) tuples for the original model and other fairness methods
baselines = {
    'default': (0.195, 0.746),
    'REW': (0.126, 0.747),
    'DIR': (0.087, 0.735),
    'Fair-Smote': (0.124, 0.719),
    'EOP': (0.141, 0.688),
    'CEO': (0.226, 0.724),
    'ROC': (0.119, 0.696),
    'MAAT': (0.131, 0.749),
    'FairMask': (0.089, 0.728)
}


# Define a function to fit and plot each HIFI curve
def plot_fitted_curve(coordinate_data, label, style, degree=5):
    # Deduplicate and sort coordinates
    unique_coords = sorted(set(coordinate_data))
    x, y = zip(*unique_coords)

    # Fit polynomial and generate smooth curve
    poly_fit = np.polyfit(x, y, degree)
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = np.polyval(poly_fit, x_smooth)

    # Plot fitted curve
    plt.plot(x_smooth, y_smooth, label=label, linestyle=style, linewidth=2.5)


# Plotting the figure
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
plt.figure(figsize=(12, 8))

# Plot each HIFI fitted curve
plot_fitted_curve(coordinate_HIFI, 'HIFI', '-', degree=5)
plot_fitted_curve(coordinate_REW_HIFI, 'REW+HIFI', '--', degree=5)
plot_fitted_curve(coordinate_MAAT_HIFI, 'MAAT+HIFI', '-.', degree=5)
plot_fitted_curve(coordinate_FairMask_HIFI, 'FairMask+HIFI', ':', degree=5)

# Plot baseline points
marker_styles = ['o', 's', '^', 'D', 'X', 'P', 'v', '<', '>', '8', 'h', 'p']
for i, (name, (x, y)) in enumerate(baselines.items()):
    plt.scatter(x, y, label=name, marker=marker_styles[i % len(marker_styles)], s=100)

# Set axis labels and legend with two columns
plt.xlabel('AOD')
plt.ylabel('Accuracy')
plt.legend(ncol=3)
plt.grid()

# Add frame lines around all sides
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)

# Show the plot
plt.savefig('NN_trade-off.png', dpi=300)
plt.show()