import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


data = {
    'Method': ['default', 'REW', 'Fair-Smote', 'MAAT', 'ADV*', 'META*', 'PR*', 'DIR', 'FairMask', 'EOP', 'CEO', 'ROC', 'HIFI*'],
    'Win-win': [24.7, 35.8, 29.5, 35.3, 36.2, 5.6, 22.2, 14.2, 35.6, 7.8, 12.7, 11.7, 37.0],
    'Good': [15.6, 46.5, 44.8, 51.5, 17.2, 13.5, 56.9, 55.4, 48.4, 45.4, 35.2, 21.0, 47.9],
    'Inverted': [26.6, 5.7, 4.5, 4.5, 28.9, 2.4, 5.0, 5.5, 8.3, 6.5, 8.8, 18.9, 4.9],
    'Poor': [7.1, 2.3, 15.6, 3.9, 1.0, 27.7, 4.5, 5.1, 3.7, 20.6, 10.3, 21.9, 5.6],
    'Lose-lose': [26.0, 9.7, 5.5, 4.7, 16.8, 50.7, 11.4, 19.8, 4.1, 19.7, 33.0, 26.5, 4.6]
}


df = pd.DataFrame(data)


df['Sum'] = df['Win-win'] + df['Good']
df = df.sort_values(by='Sum', ascending=False).drop('Sum', axis=1)


categories = df['Method']
data_values = df[['Win-win', 'Good', 'Inverted', 'Poor', 'Lose-lose']].values.T


plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})


fig, ax = plt.subplots(figsize=(12, 8))


colors = ['#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#636363']
labels = ['Win-win', 'Good', 'Inverted', 'Poor', 'Lose-lose']


for i, (colname, color) in enumerate(zip(labels, colors)):
    widths = data_values[i]
    starts = data_values[:i].sum(axis=0)
    bars = ax.bar(categories, widths, bottom=starts, label=colname, color=color)
    for bar, width in zip(bars, widths):
        if width > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f'{width}%', ha='center', va='center', fontsize=16, color='black')

ax.set_ylim(0, 100)


ax.set_ylabel('Proportions of cases in different effectiveness levels', fontsize=20)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5)


def percent(x, pos):
    return f'{int(x)}%'

ax.yaxis.set_major_formatter(FuncFormatter(percent))


plt.xticks(rotation=45)
plt.tight_layout()


plt.savefig('fairea.png', dpi=300)


plt.show()