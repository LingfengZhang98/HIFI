import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


fairness_data = {
    'Census Income': {
        'REW': (100, 100),
        'DIR': (100, 44.4),
        'Fair-Smote': (100, 88.9),
        'META*': (22.2, 22.2),
        'ADV*': (100, 44.4),
        'PR*': (66.7, 66.7),
        'EOP': (100, 100),
        'CEO': (100, 100),
        'ROC': (55.6, 55.6),
        'MAAT': (100, 100),
        'FairMask': (100, 100),
        'HIFI*': (100, 100),
    },
    'UFRGS': {
        'REW': (66.7, 66.7),
        'DIR': (100, 100),
        'Fair-Smote': (100, 100),
        'META*': (100, 100),
        'ADV*': (100, 88.9),
        'PR*': (100, 100),
        'EOP': (66.7, 66.7),
        'CEO': (33.3, 33.3),
        'ROC': (100, 88.9),
        'MAAT': (88.9, 88.9),
        'FairMask': (100, 100),
        'HIFI*': (100, 100),
    },
    'COMPAS': {
        'REW': (100, 22.2),
        'DIR': (66.7, 66.7),
        'Fair-Smote': (100, 100),
        'META*': (0, 0),
        'ADV*': (22.2, 0),
        'PR*': (100, 66.7),
        'EOP': (88.9, 0),
        'CEO': (66.7, 0),
        'ROC': (22.2, 0),
        'MAAT': (100, 55.6),
        'FairMask': (77.8, 66.7),
        'HIFI*': (100, 66.7),
    },
    'Diabetes': {
        'REW': (100, 100),
        'DIR': (100, 100),
        'Fair-Smote': (100, 100),
        'META*': (100, 66.7),
        'ADV*': (0, 0),
        'PR*': (100, 100),
        'EOP': (100, 100),
        'CEO': (0, 0),
        'ROC': (100, 100),
        'MAAT': (100, 100),
        'FairMask': (100, 66.7),
        'HIFI*': (100, 66.7),
    },
    'Default Credit': {
        'REW': (100, 100),
        'DIR': (33.3, 33.3),
        'Fair-Smote': (77.8, 11.1),
        'META*': (44.4, 33.3),
        'ADV*': (0, 0),
        'PR*': (66.7, 66.7),
        'EOP': (100, 11.1),
        'CEO': (100, 22.2),
        'ROC': (0, 0),
        'MAAT': (100, 100),
        'FairMask': (66.7, 33.3),
        'HIFI*': (100, 55.6),
    }
}

performance_data = {
    'Census Income': {
        'REW': (40, 0),
        'DIR': (80, 20),
        'Fair-Smote': (40, 40),
        'META*': (100, 100),
        'ADV*': (0, 0),
        'PR*': (100, 20),
        'EOP': (100, 100),
        'CEO': (100, 100),
        'ROC': (80, 80),
        'MAAT': (80, 80),
        'FairMask': (100, 100),
        'HIFI*': (0, 0),
    },
    'UFRGS': {
        'REW': (80, 40),
        'DIR': (100, 100),
        'Fair-Smote': (100, 100),
        'META*': (100, 100),
        'ADV*': (80, 0),
        'PR*': (100, 100),
        'EOP': (100, 100),
        'CEO': (100, 100),
        'ROC': (100, 100),
        'MAAT': (80, 40),
        'FairMask': (100, 100),
        'HIFI*': (100, 100),
    },
    'COMPAS': {
        'REW': (40, 0),
        'DIR': (100, 0),
        'Fair-Smote': (100, 40),
        'META*': (100, 100),
        'ADV*': (0, 0),
        'PR*': (40, 0),
        'EOP': (100, 0),
        'CEO': (100, 0),
        'ROC': (100, 0),
        'MAAT': (0, 0),
        'FairMask': (100, 0),
        'HIFI*': (80, 0),
    },
    'Diabetes': {
        'REW': (80, 0),
        'DIR': (80, 0),
        'Fair-Smote': (20, 0),
        'META*': (0, 0),
        'ADV*': (100, 100),
        'PR*': (100, 80),
        'EOP': (100, 100),
        'CEO': (100, 100),
        'ROC': (40, 0),
        'MAAT': (0, 0),
        'FairMask': (60, 0),
        'HIFI*': (0, 0),
    },
    'Default Credit': {
        'REW': (80, 60),
        'DIR': (100, 100),
        'Fair-Smote': (60, 60),
        'META*': (80, 80),
        'ADV*': (20, 20),
        'PR*': (80, 0),
        'EOP': (100, 40),
        'CEO': (100, 0),
        'ROC': (40, 40),
        'MAAT': (40, 0),
        'FairMask': (20, 0),
        'HIFI*': (100, 0),
    }
}

fairea_data = {
    'Census Income': {
        'REW': (98.2, 66),
        'DIR': (48.6, 16),
        'Fair-Smote': (78.9, 55.3),
        'META*': (0.9, 0),
        'ADV*': (79.4, 73.6),
        'PR*': (59.8, 20.2),
        'EOP': (81.3, 0),
        'CEO': (88.4, 2),
        'ROC': (40, 11.3),
        'MAAT': (98.9, 20),
        'FairMask': (95.8, 51.6),
        'HIFI*': (94, 47.6),
    },
    'UFRGS': {
        'REW': (69.1, 18.9),
        'DIR': (100, 0),
        'Fair-Smote': (72.4, 0),
        'META*': (26, 0),
        'ADV*': (96.5, 42.9),
        'PR*': (100, 0),
        'EOP': (33.3, 0),
        'CEO': (11.3, 0),
        'ROC': (31.1, 0),
        'MAAT': (84, 24.4),
        'FairMask': (98.4, 0),
        'HIFI*': (89.8, 4),
    },
    'COMPAS': {
        'REW': (64.7, 38),
        'DIR': (75.8, 29.6),
        'Fair-Smote': (100, 22),
        'META*': (0, 0),
        'ADV*': (41.1, 29.8),
        'PR*': (88.5, 41.8),
        'EOP': (40.2, 17.8),
        'CEO': (41.4, 23.6),
        'ROC': (30.5, 12.7),
        'MAAT': (70.5, 38.9),
        'FairMask': (80, 33.6),
        'HIFI*': (76, 36.4),
    },
    'Diabetes': {
        'REW': (100, 48),
        'DIR': (97.3, 34),
        'Fair-Smote': (97.3, 70),
        'META*': (76.7, 56),
        'ADV*': (6.7, 0),
        'PR*': (81.3, 6),
        'EOP': (69.3, 0),
        'CEO': (2, 1.3),
        'ROC': (93.3, 56),
        'MAAT': (96.7, 58.7),
        'FairMask': (79.4, 44.7),
        'HIFI*': (89.3, 69.3),
    },
    'Default Credit': {
        'REW': (91.5, 16.4),
        'DIR': (22.7, 2.7),
        'Fair-Smote': (38.4, 27.3),
        'META*': (30.7, 5.8),
        'ADV*': (12.3, 10.7),
        'PR*': (67.5, 32.2),
        'EOP': (52.9, 16.2),
        'CEO': (65.8, 29.1),
        'ROC': (9.1, 8),
        'MAAT': (90.9, 50.2),
        'FairMask': (63.3, 54.2),
        'HIFI*': (78.2, 49.1),
    }
}

methods = ['REW', 'DIR', 'Fair-Smote', 'META*', 'ADV*', 'PR*', 'EOP', 'CEO', 'ROC', 'MAAT', 'FairMask', 'HIFI*']
datasets = ['Census Income', 'UFRGS', 'COMPAS', 'Diabetes', 'Default Credit']

colors = plt.cm.get_cmap('tab20', len(methods))


plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10))


bar_width = 0.6
spacing = 1.0
bar_locs = np.arange(len(datasets))


for i, method in enumerate(methods):
    outer_bars = [fairness_data[dataset][method][0] for dataset in datasets]
    inner_bars = [fairness_data[dataset][method][1] for dataset in datasets]
    
    bar_positions = bar_locs + i * bar_width / len(methods) * spacing
    
    ax1.bar(bar_positions, outer_bars, bar_width / len(methods), edgecolor=colors(i), color='none', linestyle='--', linewidth=2)
    ax1.bar(bar_positions, inner_bars, bar_width / len(methods), color=colors(i), alpha=0.7)

ax1.set_ylim(0, 100)


def percent(x, pos):
    return f'{int(x)}%'

ax1.yaxis.set_major_formatter(FuncFormatter(percent))


ax1.set_xticks(bar_locs + bar_width / 2)
ax1.set_xticklabels(datasets)
ax1.set_ylabel('Proportions of cases of\nfairness improvement')


for i, method in enumerate(methods):
    outer_bars = [performance_data[dataset][method][0] for dataset in datasets]
    inner_bars = [performance_data[dataset][method][1] for dataset in datasets]
    
    bar_positions = bar_locs + i * bar_width / len(methods) * spacing
    
    ax2.bar(bar_positions, outer_bars, bar_width / len(methods), edgecolor=colors(i), color='none', linestyle='--', linewidth=2)
    ax2.bar(bar_positions, inner_bars, bar_width / len(methods), color=colors(i), alpha=0.7)

ax2.set_ylim(0, 100)

ax2.yaxis.set_major_formatter(FuncFormatter(percent))


ax2.set_xticks(bar_locs + bar_width / 2)
ax2.set_xticklabels(datasets)
ax2.set_ylabel('Proportions of cases of\nperformance degradation')


for i, method in enumerate(methods):
    outer_bars = [fairea_data[dataset][method][0] for dataset in datasets]
    inner_bars = [fairea_data[dataset][method][1] for dataset in datasets]
    
    bar_positions = bar_locs + i * bar_width / len(methods) * spacing
    
    ax3.bar(bar_positions, outer_bars, bar_width / len(methods), edgecolor=colors(i), color='none', linestyle='--', linewidth=2)
    ax3.bar(bar_positions, inner_bars, bar_width / len(methods), color=colors(i), alpha=0.7)

ax3.set_ylim(0, 100)

ax3.yaxis.set_major_formatter(FuncFormatter(percent))


ax3.set_xticks(bar_locs + bar_width / 2)
ax3.set_xticklabels(datasets)
ax3.set_ylabel("Proportions of cases of\n'win-win' and 'good'")


legend_elements = [Patch(facecolor=colors(i), edgecolor=colors(i), linestyle='--', label=method) for i, method in enumerate(methods)]
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=5)

plt.tight_layout()
plt.savefig('dataset_wise.png', dpi=300)
plt.show()