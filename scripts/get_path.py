import numpy as np

# Load data
data = np.load("drone_paths_final_file.npz", allow_pickle=True)

agent0 = data["agent0"].item()
episodes = data["episode"]
starts = agent0["start"]
goals = agent0["goal"]
final_path = agent0["final_path"]
obstacles = data["obstacles"]

# Select episode number to view

episode_number = 4905


start_episode = episodes[0]  # should be 4651
index = episode_number - start_episode

# Extract data
start = starts[index]
goal = goals[index]
path = final_path[index]
obs = obstacles[index]

print(start)
print(goal)
print(path)