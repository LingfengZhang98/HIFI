"""
We use experimental results on LR models here.
"""

import matplotlib.pyplot as plt
import numpy as np


# (AOD, accuracy) tuples for HIFI with eta in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100, 1000]
coordinate_HIFI = [
    (0.205, 0.740),
    (0.198, 0.738),
    (0.150, 0.744),
    (0.107, 0.743),
    (0.102, 0.744),
    (0.095, 0.742),
    (0.106, 0.743),
    (0.100, 0.742),
    (0.091, 0.740),
    (0.094, 0.745),
    (0.057, 0.631),
    (0.051, 0.599)
]

# (AOD, accuracy) tuples for the original model and other fairness methods
baselines = {
    'default': (0.200, 0.749),
    'REW': (0.138, 0.749),
    'DIR': (0.094, 0.739),
    'Fair-Smote': (0.084, 0.720),
    'EOP': (0.153, 0.718),
    'CEO': (0.219, 0.724),
    'ROC': (0.158, 0.710),
    'MAAT': (0.148, 0.749),
    'FairMask': (0.110, 0.744),
    'META*': (0.229, 0.701),
    'ADV*': (0.196, 0.705),
    'PR*': (0.123, 0.729)
}

# Extract and deduplicate HIFI coordinates
unique_hifi_coords = sorted(set(coordinate_HIFI))
hifi_x_unique, hifi_y_unique = zip(*unique_hifi_coords)

# Fit a 5th-degree polynomial
quintic_fit = np.polyfit(hifi_x_unique, hifi_y_unique, 5)
hifi_x_poly_smooth = np.linspace(min(hifi_x_unique), max(hifi_x_unique), 500)
hifi_y_quintic_smooth = np.polyval(quintic_fit, hifi_x_poly_smooth)

# Plotting
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
plt.figure(figsize=(12, 8))

# Plot the 5th-degree polynomial fit curve with increased line width
plt.plot(hifi_x_poly_smooth, hifi_y_quintic_smooth, label='HIFI*', linestyle='-', linewidth=2.5)

# Plot baseline method points with increased size
marker_styles = ['o', 's', '^', 'D', 'X', 'P', 'v', '<', '>', 'h', '^', 'p']
for i, (name, (x, y)) in enumerate(baselines.items()):
    plt.scatter(x, y, label=name, marker=marker_styles[i % len(marker_styles)], s=100)  # Increase 's' for point size

# Set axis labels and legend
plt.xlabel('AOD')
plt.ylabel('Accuracy')
plt.legend(ncol=2)
plt.grid()

# Add frame lines around all sides
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)

# Show the plot
plt.savefig('LR_trade-off.png', dpi=300)
plt.show()