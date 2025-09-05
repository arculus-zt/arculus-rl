import numpy as np

# Load cleaned data
data = np.load("drone_paths_final_file.npz", allow_pickle=True)

agent0 = data["agent0"].item()
episodes = data["episode"]
starts = agent0["start"]
goals = agent0["goal"]
paths = agent0["final_path"]

# Compute distances and collect all relevant data
entries = []
for ep, start, goal, path in zip(episodes, starts, goals, paths):
    distance = np.linalg.norm(np.array(goal) - np.array(start))  # Euclidean
    entries.append((ep, start, goal, path, distance))

# Sort entries by distance descending
entries_sorted = sorted(entries, key=lambda x: x[-1], reverse=True)

# Example: print top 5
print("Top n episodes with longest start-goal distance:")
for ep, start, goal, path, dist in entries_sorted[:50]:
    print(f"Episode {ep}: Distance = {dist:.2f}, Start = {start}, Goal = {goal}, Path Length = {len(path)}")