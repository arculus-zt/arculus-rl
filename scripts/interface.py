"""
interface.py
============

Example
-------
>>> start      = (0, 0)
>>> goal       = (14, 14)
>>> obstacles  = [(2,3), (7,8), (11,1)]          # or [] for none
>>> path = get_path(start, goal, obstacles)
>>> for i, p in enumerate(path):
...     print(f"step {i}: {p}")
"""
import torch
from gridworld import Gridworld, DQN


def _prepare_env(start, goal, obstacles):
    """
    Instantiate Gridworld, then hard-set the first agent’s start/goal/obstacles.
    The second agent is parked at an unused corner so its reward terms are silent.
    """
    env = Gridworld()

    # lock positions/obstacles
    env.agents          = [start, (0, 0)]          # agent-1 dummy
    env.goals           = [goal,  (0, 0)]          # goal-1 dummy
    env.obstacles       = list(obstacles)          # make sure it’s a list

    # rebuild A* helpers for shaping/reward
    env.best_path_agent0 = env.a_star(start, goal)
    env.best_path_agent1 = []                      # not used

    return env


def _load_model(state_dim, model_path="agent_0_trained.pth"):
    model = DQN(state_dim, 4)          # 4 actions: N,S,W,E
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def get_path(start, goal, obstacles, model_path="agent_0_trained.pth",
             max_steps=500):
    """
    Parameters
    ----------
    start, goal : tuple[int, int]
        Grid coordinates (x, y).
    obstacles   : list[tuple[int,int]]
    model_path  : str
    max_steps   : int
    Returns
    -------
    list[tuple[int,int]]
        Visited (x,y) cells, starting with `start`.
    """
    env   = _prepare_env(start, goal, obstacles)
    state = env.get_states()[0]                    # only agent-0’s state
    net   = _load_model(len(state), model_path)

    path, done = [start], False
    while not done and len(path) < max_steps:
        with torch.no_grad():
            action = net(torch.tensor(state, dtype=torch.float32)
                         .unsqueeze(0)).argmax().item()

        # second agent gets a no-op               ↓↓↓
        next_states, _, done = env.step([action, 0])
        state = next_states[0]
        path.append(env.agents[0])

    return path


if __name__ == "__main__":
    # quick sanity check
    demo_start = (0, 0)
    demo_goal  = (14, 14)
    demo_obs   = [(2, 3), (7, 8), (11, 1), (6, 4), (9, 13),(3, 10), (12, 7), (0, 9), (0, 12), (4, 6),(10, 3), (1, 10), (6, 9), (13, 2), (7, 6),(5, 14), (11, 12), (10, 14), (8, 2), (14, 4),(5, 8), (2, 10), (13, 13), (10, 9), (4, 2)]

    final_path = get_path(demo_start, demo_goal, demo_obs)

    print("\nFINAL PATH")
    for i, cell in enumerate(final_path):
        print(f"step {i:<3}: {cell}")
