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

# Filter and print top entries with path length < 100
print("Top episodes with longest start-goal distance and path length < 100:")
count = 0
for ep, start, goal, path, dist in entries_sorted:
    if len(path) < 100:
        print(f"Episode {ep}: Distance = {dist:.2f}, Start = {start}, Goal = {goal}, Path Length = {len(path)}")
        count += 1
    if count == 50:
        break
