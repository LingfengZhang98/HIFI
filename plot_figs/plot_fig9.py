import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import FuncFormatter


file_path = 'data4fig9.txt'
with open(file_path, 'r') as file:
    content = file.read()


adjusted_content = re.sub(r'\t+', '\t', content)
adjusted_content = re.sub(r'%', '', adjusted_content)


adjusted_file_path = 'adjusted_causal_fairness.txt'
with open(adjusted_file_path, 'w') as file:
    file.write(adjusted_content)


data = pd.read_csv(adjusted_file_path, sep='\t', header=None)


data.columns = ['Method', 'First_Column', 'Second_Column']


data['First_Column'] = data['First_Column'].astype(float)
data['Second_Column'] = data['Second_Column'].astype(float)

plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})


fig, ax1 = plt.subplots(figsize=(12, 5))

def percent(x, pos):
    return f'{int(x)}%'

ax1.yaxis.set_major_formatter(FuncFormatter(percent))

color = 'tab:blue'
ax1.set_ylabel('CFVR', color=color, fontsize=22)
ax1.plot(data['Method'], data['First_Column'], color=color, marker='o', label='First_Column')
ax1.tick_params(axis='y', labelcolor=color)


ax1.set_xticks(data['Method'])
ax1.set_xticklabels(data['Method'], rotation=45, ha='right')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Average purely sensitive\ninteraction strength', color=color, fontsize=22)
ax2.plot(data['Method'], data['Second_Column'], color=color, marker='x', label='Second_Column')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig('causal_fairness.png', dpi=300)
plt.show()