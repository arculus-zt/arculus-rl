import matplotlib.pyplot as plt
import pandas as pd


# Data for the three battery levels
data = {
    'Reward Ratio': list(range(11)),
    'Battery 50000': {
        'Safety': [8568, 7327, 6996, 5911, 3713, 2041, 1923, 1497, 1245, 1291, 1105],
        'Success': [0, 7, 13, 13, 13, 21, 36, 18, 20, 20, 37],
        'Success but dead': [0, 967, 1641, 2797, 5131, 6893, 7141, 7496, 7873, 7824, 8078],
        'Both Failed': [1432, 1699, 1350, 1279, 1135, 1039, 900, 989, 862, 865, 780]
    },
    'Battery 100000': {
        'Safety': [9949, 9125, 8012, 6073, 3858, 3100, 2325, 1939, 1707, 943, 920],
        'Success': [0, 875, 1988, 3927, 6142, 6900, 7675, 8061, 8293, 9057, 9080],
        'Success but dead': [0, 279, 581, 551, 989, 1037, 964, 1016, 941, 943, 920],
        'Both Failed': [51, 196, 149, 149, 136, 140, 117, 107, 118, 99, 90]
    },
    'Battery 500000': {
        'Safety': [9937, 9945, 9024, 7551, 6394, 3008, 2479, 2366, 1759, 1800, 1382],
        'Success': [0, 55, 976, 2449, 3606, 6992, 7521, 7634, 8241, 8200, 8618],
        'Success but dead': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Both Failed': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
}

# Convert values to percentages
for key, battery_level in data.items():
    if key != 'Reward Ratio':  # Skip the Reward Ratio list
        for metric in battery_level.keys():
            battery_level[metric] = [val / 100 for val in battery_level[metric]]

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'Reward Ratio': data['Reward Ratio'],
    'Battery 50000 - Safety': data['Battery 50000']['Safety'],
    'Battery 50000 - Success': data['Battery 50000']['Success'],
    'Battery 50000 - Success but Dead': data['Battery 50000']['Success but dead'],
    'Battery 50000 - Both Failed': data['Battery 50000']['Both Failed'],
    'Battery 100000 - Safety': data['Battery 100000']['Safety'],
    'Battery 100000 - Success': data['Battery 100000']['Success'],
    'Battery 100000 - Success but Dead': data['Battery 100000']['Success but dead'],
    'Battery 100000 - Both Failed': data['Battery 100000']['Both Failed'],
    'Battery 500000 - Safety': data['Battery 500000']['Safety'],
    'Battery 500000 - Success': data['Battery 500000']['Success'],
    'Battery 500000 - Success but Dead': data['Battery 500000']['Success but dead'],
    'Battery 500000 - Both Failed': data['Battery 500000']['Both Failed'],
})

# Define color codes
color_safety = 'blue'
color_success = 'green'
color_success_but_dead = 'orange'
color_both_failed = 'red'

# Plot the data
plt.figure(figsize=(14, 8))

marker_size = 10

# Plot for Safety
plt.plot(df['Reward Ratio'], df['Battery 50000 - Safety'], label='Battery 50000 - Safety', color=color_safety, linestyle='--', marker='o', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 100000 - Safety'], label='Battery 100000 - Safety', color=color_safety, linestyle='--', marker='s', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 500000 - Safety'], label='Battery 500000 - Safety', color=color_safety, linestyle='--', marker='^', markersize=marker_size)

# Plot for Success
plt.plot(df['Reward Ratio'], df['Battery 50000 - Success'], label='Battery 50000 - Success', color=color_success, linestyle='-', marker='o', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 100000 - Success'], label='Battery 100000 - Success', color=color_success, linestyle='-', marker='s', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 500000 - Success'], label='Battery 500000 - Success', color=color_success, linestyle='-', marker='^', markersize=marker_size)

# Plot for Success but Dead
plt.plot(df['Reward Ratio'], df['Battery 50000 - Success but Dead'], label='Battery 50000 - Success but Died', color=color_success_but_dead, linestyle='-.', marker='o', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 100000 - Success but Dead'], label='Battery 100000 - Success but Died', color=color_success_but_dead, linestyle='-.', marker='s', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 500000 - Success but Dead'], label='Battery 500000 - Success but Died', color=color_success_but_dead, linestyle='-.', marker='^', markersize=marker_size)

# Plot for Both Failed
plt.plot(df['Reward Ratio'], df['Battery 50000 - Both Failed'], label='Battery 50000 - Both Failed', color=color_both_failed, linestyle=':', marker='o', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 100000 - Both Failed'], label='Battery 100000 - Both Failed', color=color_both_failed, linestyle=':', marker='s', markersize=marker_size)
plt.plot(df['Reward Ratio'], df['Battery 500000 - Both Failed'], label='Battery 500000 - Both Failed', color=color_both_failed, linestyle=':', marker='^', markersize=marker_size)

# Customize the plot



plt.xlabel('Reward Ratio', fontsize=20)
plt.xticks(fontsize=18)
plt.ylabel('Percentage (%)', fontsize=20)
plt.yticks(fontsize=18)
plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.grid(False)
plt.tight_layout()

# Show the plot
plt.show()
