import matplotlib.pyplot as plt
import pandas as pd

# Data for the trust zone radii
data = {
    'Reward Ratio': list(range(11)),
    'Radius 0': {
        'Safety': [10000, 9923, 8946, 7879, 4819, 2658, 2230, 2116, 1892, 1802, 1756],
        'Success': [0, 77, 1054, 2121, 5181, 7342, 7770, 7884, 8108, 8498, 8520],
    },
    'Radius 250': {
        'Safety': [10000, 9934, 9056, 7407, 4103, 2550, 2370, 2054, 1915, 1672, 1487],
        'Success': [0, 66, 771, 1593, 6280, 7119, 7630, 7946, 8085, 8328, 8513],
    },
    'Radius 500': {
        'Safety': [10000, 9946, 9226, 6720, 3956, 2370, 2407, 2032, 1922, 1589, 1336],
        'Success': [0, 54, 944, 3280, 4982, 7031, 7593, 7968, 8078, 8411, 8401],
    },
}

# Convert values to percentages
for radius, values in data.items():
    if radius != 'Reward Ratio':  
        for metric in values.keys():
            values[metric] = [val / 100 for val in values[metric]]

# Create DataFrame
df = pd.DataFrame({
    'Reward Ratio': data['Reward Ratio'],
    'Radius 0 - Safety': data['Radius 0']['Safety'],
    'Radius 0 - Success': data['Radius 0']['Success'],
    'Radius 250 - Safety': data['Radius 250']['Safety'],
    'Radius 250 - Success': data['Radius 250']['Success'],
    'Radius 500 - Safety': data['Radius 500']['Safety'],
    'Radius 500 - Success': data['Radius 500']['Success'],
})

# Define color codes
color_safety = 'blue'
color_success = 'green'

plt.figure(figsize=(14, 8))

marker_size = 10

# Safety plots
plt.plot(df['Reward Ratio'], df['Radius 0 - Safety'], label='Radius 0 - Safety', color=color_safety, linestyle='--', marker='o', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Radius 250 - Safety'], label='Radius 250 - Safety', color=color_safety, linestyle='--', marker='s', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Radius 500 - Safety'], label='Radius 500 - Safety', color=color_safety, linestyle='--', marker='^', markersize=marker_size)

# Success plots
plt.plot(df['Reward Ratio'], df['Radius 0 - Success'], label='Radius 0 - Success', color=color_success, linestyle='-', marker='o', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Radius 250 - Success'], label='Radius 250 - Success', color=color_success, linestyle='-', marker='s', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Radius 500 - Success'], label='Radius 500 - Success', color=color_success, linestyle='-', marker='^', markersize=marker_size)

# Axis labels with bold + Arial
plt.xlabel('Reward Ratio', fontsize=20, fontweight='bold', fontname='Arial')
plt.ylabel('Percentage (%)', fontsize=20, fontweight='bold', fontname='Arial')

# Ticks
plt.xticks(fontsize=18, fontname='Arial')
plt.yticks(fontsize=18, fontname='Arial')

# Legend with Arial
plt.legend(prop={'family': 'Arial', 'size': 16})




plt.grid(False)
plt.tight_layout()

plt.show()
