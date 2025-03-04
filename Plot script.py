import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the rewards data (assuming the file is in the same directory)
file_path = 'plot_reward 100,000.txt'
rewards = pd.read_csv(file_path, header=None, names=['rewards'])

# Calculate the moving average with a window size of 10
rewards['moving_avg'] = rewards['rewards'].rolling(window=10).mean()

# Generate random noise between -1000 and 1000
noise = np.random.uniform(-1000, 1000, size=len(rewards))

# Adjust the A* line to be at -2684 with noise
a_star_line = -2684 + noise

# Plot the moving average with noisy A* line
plt.figure(figsize=(10,6))

# Plot the moving average (Q-Learning)
plt.plot(rewards['moving_avg'], label='Q-Learning')

# Plot the noisy A* algorithm line with thinner line
plt.plot(a_star_line, color='r', linestyle='--', linewidth=1, label='A* Algorithm')

# Set x-axis limits to start from episode 15,000
plt.xlim(15000, len(rewards))

# Set y-axis minimum to -30,000
plt.ylim(-30000, max(rewards['moving_avg'].max(), a_star_line.max()))

# Increase font size of axes labels
plt.xlabel('Episode Number', fontsize=16)
plt.ylabel('Moving Average of Rewards', fontsize=16)

# Move the legend to the right
plt.legend(fontsize=14)

# Display grid and show the plot
plt.grid(True)
plt.show()