import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


fairness_data = {
    'LR': {
        'REW': (92.3, 74.4),
        'DIR': (76.9, 64.1),
        'Fair-Smote': (94.9, 76.9),
        'META*': (46.2, 41.0),
        'ADV*': (51.3, 30.8),
        'PR*': (84.6, 76.9),
        'EOP': (89.7, 48.7),
        'CEO': (69.2, 35.9),
        'ROC': (48.7, 41.0),
        'MAAT': (97.4, 87.2),
        'FairMask': (87.2, 74.4),
        'HIFI*': (100, 79.5),
    },
    'SVM': {
        'REW': (89.7, 53.8),
        'DIR': (74.4, 61.5),
        'Fair-Smote': (69.2, 69.2),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (69.2, 43.6),
        'CEO': (43.6, 23.1),
        'ROC': (51.3, 41.0),
        'MAAT': (94.9, 61.5),
        'FairMask': (87.2, 71.8),
        'HIFI*': (76.9, 66.7),
    },
    'NN': {
        'REW': (94.9, 33.3),
        'DIR': (69.2, 48.7),
        'Fair-Smote': (61.5, 53.8),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (84.6, 28.2),
        'CEO': (71.8, 17.9),
        'ROC': (56.4, 38.5),
        'MAAT': (94.9, 43.6),
        'FairMask': (94.9, 56.4),
        'HIFI*': (51.3, 23.1),
    },
    'RF': {
        'REW': (76.9, 12.8),
        'DIR': (64.1, 35.9),
        'Fair-Smote': (74.4, 33.3),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (38.5, 5.1),
        'CEO': (41.0, 5.1),
        'ROC': (41.0, 28.2),
        'MAAT': (97.4, 56.4),
        'FairMask': (82.1, 38.5),
        'HIFI*': 'X',
    }
}

performance_data = {
    'LR': {
        'REW': (64, 20),
        'DIR': (92, 44),
        'Fair-Smote': (64, 48),
        'META*': (76, 76),
        'ADV*': (40, 24),
        'PR*': (84, 40),
        'EOP': (100, 68),
        'CEO': (100, 60),
        'ROC': (72, 44),
        'MAAT': (40, 24),
        'FairMask': (68, 20),
        'HIFI*': (56, 20),
    },
    'SVM': {
        'REW': (68, 24),
        'DIR': (92, 44),
        'Fair-Smote': (64, 48),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (84, 64),
        'CEO': (84, 60),
        'ROC': (80, 48),
        'MAAT': (36, 24),
        'FairMask': (68, 20),
        'HIFI*': (64, 16),
    },
    'NN': {
        'REW': (68, 16),
        'DIR': (76, 36),
        'Fair-Smote': (60, 36),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (100, 68),
        'CEO': (100, 52),
        'ROC': (72, 44),
        'MAAT': (64, 20),
        'FairMask': (80, 40),
        'HIFI*': (40, 16),
    },
    'RF': {
        'REW': (56, 0),
        'DIR': (80, 68),
        'Fair-Smote': (44, 20),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (60, 20),
        'CEO': (80, 12),
        'ROC': (68, 60),
        'MAAT': (64, 12),
        'FairMask': (100, 40),
        'HIFI*': 'X',
    }
}

fairea_data = {
    'LR': {
        'REW': (82.3, 35.8),
        'DIR': (69.6, 14.2),
        'Fair-Smote': (74.3, 29.5),
        'META*': (19.1, 5.6),
        'ADV*': (53.4, 36.2),
        'PR*': (79.1, 22.2),
        'EOP': (53.2, 7.8),
        'CEO': (47.9, 12.7),
        'ROC': (32.7, 11.7),
        'MAAT': (86.8, 35.3),
        'FairMask': (84, 35.6),
        'HIFI*': (84.9, 37),
    },
    'SVM': {
        'REW': (75.5, 31.1),
        'DIR': (67, 14.3),
        'Fair-Smote': (67.5, 22.1),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (45.6, 11.3),
        'CEO': (36.5, 11.9),
        'ROC': (31.1, 9.5),
        'MAAT': (76.4, 36),
        'FairMask': (79.8, 32.4),
        'HIFI*': (77.4, 30.8),
    },
    'NN': {
        'REW': (73.2, 37.5),
        'DIR': (65, 19.9),
        'Fair-Smote': (50.9, 15.6),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (59.4, 24.1),
        'CEO': (50.8, 22.9),
        'ROC': (53.8, 22.6),
        'MAAT': (80.6, 39.7),
        'FairMask': (80.4, 26.5),
        'HIFI*': (52.1, 22.9),
    },
    'RF': {
        'REW': (57.9, 31.6),
        'DIR': (54.5, 16.3),
        'Fair-Smote': (48.1, 32.3),
        'META*': 'X',
        'ADV*': 'X',
        'PR*': 'X',
        'EOP': (43.9, 18),
        'CEO': (38.1, 16.6),
        'ROC': (40.5, 10.9),
        'MAAT': (78.9, 39.3),
        'FairMask': (66.9, 16.5),
        'HIFI*': 'X',
    }
}

methods = ['REW', 'DIR', 'Fair-Smote', 'META*', 'ADV*', 'PR*', 'EOP', 'CEO', 'ROC', 'MAAT', 'FairMask', 'HIFI*']
models = ['SVM', 'NN', 'RF']

colors = plt.cm.get_cmap('tab20', len(methods))


plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10))


bar_width = 0.6
spacing = 1.0
bar_locs = np.arange(len(models))


for i, method in enumerate(methods):
    outer_bars = [fairness_data[model][method][0] if fairness_data[model][method] != 'X' else 0 for model in models]
    inner_bars = [fairness_data[model][method][1] if fairness_data[model][method] != 'X' else 0 for model in models]
    
    bar_positions = bar_locs + i * bar_width / len(methods) * spacing
    
    ax1.bar(bar_positions, outer_bars, bar_width / len(methods), edgecolor=colors(i), color='none', linestyle='--', linewidth=2)
    bars = ax1.bar(bar_positions, inner_bars, bar_width / len(methods), color=colors(i), alpha=0.7)

    for j, val in enumerate(outer_bars):
        if fairness_data[models[j]][method] == 'X':
            ax1.text(bar_positions[j], val + 1, 'X', ha='center', va='bottom', fontsize=12, color='black')

ax1.set_ylim(0, 100)


def percent(x, pos):
    return f'{int(x)}%'

ax1.yaxis.set_major_formatter(FuncFormatter(percent))


ax1.set_xticks(bar_locs + bar_width / 2)
ax1.set_xticklabels(models)
ax1.set_ylabel('Proportions of cases of\nfairness improvement')


for i, method in enumerate(methods):
    outer_bars = [performance_data[model][method][0] if performance_data[model][method] != 'X' else 0 for model in models]
    inner_bars = [performance_data[model][method][1] if performance_data[model][method] != 'X' else 0 for model in models]
    
    bar_positions = bar_locs + i * bar_width / len(methods) * spacing
    
    ax2.bar(bar_positions, outer_bars, bar_width / len(methods), edgecolor=colors(i), color='none', linestyle='--', linewidth=2)
    bars = ax2.bar(bar_positions, inner_bars, bar_width / len(methods), color=colors(i), alpha=0.7)

    for j, val in enumerate(outer_bars):
        if performance_data[models[j]][method] == 'X':
            ax2.text(bar_positions[j], val + 1, 'X', ha='center', va='bottom', fontsize=12, color='black')

ax2.set_ylim(0, 100)

ax2.yaxis.set_major_formatter(FuncFormatter(percent))


ax2.set_xticks(bar_locs + bar_width / 2)
ax2.set_xticklabels(models)
ax2.set_ylabel('Proportions of cases of\nperformance degradation')


for i, method in enumerate(methods):
    outer_bars = [fairea_data[model][method][0] if fairea_data[model][method] != 'X' else 0 for model in models]
    inner_bars = [fairea_data[model][method][1] if fairea_data[model][method] != 'X' else 0 for model in models]
    
    bar_positions = bar_locs + i * bar_width / len(methods) * spacing
    
    ax3.bar(bar_positions, outer_bars, bar_width / len(methods), edgecolor=colors(i), color='none', linestyle='--', linewidth=2)
    bars = ax3.bar(bar_positions, inner_bars, bar_width / len(methods), color=colors(i), alpha=0.7)

    for j, val in enumerate(outer_bars):
        if fairea_data[models[j]][method] == 'X':
            ax3.text(bar_positions[j], val + 1, 'X', ha='center', va='bottom', fontsize=12, color='black')

ax3.set_ylim(0, 100)

ax3.yaxis.set_major_formatter(FuncFormatter(percent))


ax3.set_xticks(bar_locs + bar_width / 2)
ax3.set_xticklabels(models)
ax3.set_ylabel("Proportions of cases of\n'win-win' and 'good'")


legend_elements = [Patch(facecolor=colors(i), edgecolor=colors(i), linestyle='--', label=method) for i, method in enumerate(methods)]
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=5)

plt.tight_layout()
plt.savefig('model_wise.png', dpi=300)
plt.show()