# Arculus-RL: Multi-Agent Drone Pathfinding with DQN

This repository contains the **Reinforcement Learning (RL) module** for the [Arculus Project](https://github.com/arculus-zt), focused on tactical drone navigation in dynamic environments with jamming, congestion, and obstacles.

Using a multi-agent Deep Q-Network (DQN) approach, the framework trains two drones to navigate from random start points to random goals, avoiding both static obstacles and dynamic congestion zones in a gridworld.

---

## Key Features

- **Double DQN Agents**: Trained to reach individual goals with shared environment dynamics
- **Obstacle and Congestion Awareness**: Custom observation space includes:
  - Local obstacle and congestion maps
  - Direction vectors to the goal
  - Local view of the A* optimal path
- **Pygame Visualizer**: Renders training episodes as interactive videos
- **Dynamic Congestion Zones**: Simulate changing network or threat conditions
- **Replay Buffer**: Implements experience replay for stable DQN learning
- **Video Export**: Saves milestone episodes as `.mp4` files
- **Last 350-Episode Path Logging**: Logs agent behavior for post-training analysis

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/arculus-zt/arculus-rl.git
cd arculus-rl
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> Requirements include: `torch`, `pygame`, `opencv-python`, `numpy`

---

## Running the Training

```bash
python train.py
```

Training runs for 5000 episodes by default. Two agents are trained in parallel to reach their goals from randomly initialized start positions. Environment includes:

- A 15x15 grid
- Randomly placed obstacles
- Dynamic congestion zones
- Per-step shaped rewards

Models and logs are saved at the end of training:
- `agent_0_trained.pth`, `agent_1_trained.pth`
- `drone_paths_last350episodes.npz`

---

## Code Structure

```
arculus-rl/
├── train.py               # Main training loop
├── agent.py               # DQN model and agent logic (embedded in train.py)
├── envs/                  # Gridworld environment (currently part of train.py)
├── requirements.txt
└── README.md
```

---

## Observation Space Per Agent

Each agent observes:
- Its own position and goal (4 values)
- Direction vector to the goal (2 values)
- Flattened 11x11 grid around it for:
  - Obstacles (121 values)
  - Congestion zones (121 values)
  - A* path presence (121 values)

> **Total observation dimension per agent**: 369

---

## Reward Structure

| Event                             | Reward  |
|----------------------------------|---------|
| Reaching goal                    | +10,000 |
| Each step taken                  | -1      |
| Collision with wall/obstacle     | -500    |
| Entering congestion zone         | -20     |
| Getting close to the other drone | -10     |
| Following A* path                | +5      |
| Deviating far from path          | -5      |
| Moving closer to goal            | +0.5 × distance delta |

---

## Sample Milestone Rendering (Optional)

To render specific milestone episodes, add the desired episode number to the `milestones` dictionary inside `train()`:

```python
milestones = {
    500: True,
    1000: True,
    2000: True
}
```

---

## License

MIT License. Feel free to use and modify.

---

## Author

**Subrahmanya Chandra Bhamidipati**  
Graduate Researcher, Arculus-ZT Project  
GitHub: [@bvlsc](https://github.com/bvlsc)
