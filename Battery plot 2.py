import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Reward Ratio': list(range(11)),
    'Battery 50000': {
        'Returned Only': [8520, 7280, 6935, 5780, 3600, 1985, 1850, 1425, 1190, 1250, 1050],
        'Full Success': [0, 5, 10, 15, 18, 30, 40, 25, 30, 35, 50],
        'Mission Only': [0, 960, 1600, 2840, 5235, 6920, 7200, 7520, 7925, 7790, 8050],
        'Failure': [1480, 1755, 1455, 1365, 1147, 1065, 910, 1030, 855, 925, 850]
    },
    'Battery 100000': {
        'Returned Only': [9920, 9100, 7980, 6005, 3750, 3050, 2280, 1880, 1655, 985, 950],
        'Full Success': [0, 880, 1990, 3995, 6250, 7020, 7750, 8090, 8350, 9060, 9110],
        'Mission Only': [0, 290, 590, 540, 980, 990, 950, 990, 930, 960, 890],
        'Failure': [80, 230, 140, 150, 120, 140, 120, 140, 125, 95, 80]
    },
    'Battery 500000': {
        'Returned Only': [9930, 9935, 8990, 7480, 6355, 2980, 2420, 2340, 1785, 1790, 1345],
        'Full Success': [0, 60, 990, 2495, 3690, 7090, 7580, 7700, 8290, 8270, 8660],
        'Mission Only': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Failure': [70, 5, 20, 25, 55, 30, 0, 0, 0, 0, 0]
    }
}

plt.figure(figsize=(10, 6))
reward_ratios = data['Reward Ratio']

styles = {
    'Battery 50000': 'o',
    'Battery 100000': 's',
    'Battery 500000': '^'
}

colors = {
    'Returned Only': '#1f77b4',
    'Full Success': '#2ca02c',
    'Mission Only': '#ff7f0e',
    'Failure': '#d62728'
}

for category in ['Returned Only', 'Full Success', 'Mission Only', 'Failure']:
    for battery in ['Battery 50000', 'Battery 100000', 'Battery 500000']:
        label = f"{battery.split()[1]} - {category}"
        plt.plot(
            reward_ratios,
            data[battery][category],
            label=label,
            color=colors[category],
            linestyle='-' if category in ['Full Success', 'Mission Only'] else '--',
            marker=styles[battery],
            markersize=7,
            linewidth=2
        )

# Bold axis labels with Arial
plt.xlabel('Reward Ratio', fontsize=16, fontweight='bold', fontname='Arial')
plt.ylabel('Outcome Percentage (%)', fontsize=16, fontweight='bold', fontname='Arial')

# Ticks with Arial
plt.xticks(fontsize=16, fontname='Arial')
plt.yticks(fontsize=16, fontname='Arial')

plt.grid(True, linestyle=':', alpha=0.5)

# Legend with Arial
plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5), title='Battery & Outcome', title_fontsize=12, prop={'family': 'Arial'})

plt.tight_layout(rect=[0, 0, 0.82, 1])

plt.show()