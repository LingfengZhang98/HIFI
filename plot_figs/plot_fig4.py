import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


data = {
    'Method': ['REW', 'Fair-Smote', 'MAAT', 'ADV*', 'META*', 'PR*', 'DIR', 'FairMask', 'EOP', 'CEO', 'ROC', 'HIFI*'],
    'Significant Increase': [74.4, 76.9, 87.2, 30.8, 41.0, 76.9, 64.1, 74.4, 48.7, 35.9, 41.0, 79.5],
    'Slight Increase': [17.9, 18.0, 10.2, 20.5, 5.2, 7.7, 12.8, 12.8, 41.0, 33.3, 7.7, 20.5],
    'Slight Decrease': [2.6, 5.1, 2.6, 23.1, 7.6, 10.3, 7.7, 10.2, 2.6, 12.9, 25.7, 0.0],
    'Significant Decrease': [5.1, 0.0, 0.0, 25.6, 46.2, 5.1, 15.4, 2.6, 7.7, 17.9, 25.6, 0.0]
}


df = pd.DataFrame(data)


df = df.sort_values(by='Significant Increase', ascending=False)


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


ax.set_ylabel('Proportions of cases in different fairness levels', fontsize=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)


def percent(x, pos):
    return f'{int(x)}%'

ax.yaxis.set_major_formatter(FuncFormatter(percent))


plt.xticks(rotation=45)
plt.tight_layout()


plt.savefig('fairness_statistics.png', dpi=300)


plt.show()