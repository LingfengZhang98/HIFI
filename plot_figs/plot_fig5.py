import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


data = {
    'Method': ['REW', 'Fair-Smote', 'MAAT', 'ADV*', 'META*', 'PR*', 'DIR', 'FairMask', 'EOP', 'CEO', 'ROC', 'HIFI*'],
    'Significant Increase': [12, 28, 4, 36, 0, 0, 0, 12, 0, 0, 16, 0],
    'Slight Increase': [24, 8, 56, 24, 24, 16, 8, 20, 0, 0, 12, 44],
    'Slight Decrease': [44, 16, 16, 16, 0, 44, 48, 48, 32, 40, 28, 36],
    'Significant Decrease': [20, 48, 24, 24, 76, 40, 44, 20, 68, 60, 44, 20]
}


df = pd.DataFrame(data)


df = df.sort_values(by='Significant Decrease')


categories = df['Method']
data_values = df[['Significant Increase', 'Slight Increase', 'Slight Decrease', 'Significant Decrease']].values.T


plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})


fig, ax = plt.subplots(figsize=(12, 8))


colors = ['#d9d9d9', '#bdbdbd', '#969696', '#636363']
labels = ['Significant Increase', 'Slight Increase', 'Slight Decrease', 'Significant Decrease']


for i, (colname, color) in enumerate(zip(labels, colors)):
    widths = data_values[i]
    starts = data_values[:i].sum(axis=0)
    bars = ax.bar(categories, widths, bottom=starts, label=colname, color=color)
    for bar, width in zip(bars, widths):
        if width > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f'{width}%', ha='center', va='center', fontsize=16, color='black')

ax.set_ylim(0, 100)


ax.set_ylabel('Proportions of cases in different performance levels', fontsize=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)


def percent(x, pos):
    return f'{int(x)}%'

ax.yaxis.set_major_formatter(FuncFormatter(percent))


plt.xticks(rotation=45)
plt.tight_layout()


plt.savefig('performance_statistics.png', dpi=300)


plt.show()