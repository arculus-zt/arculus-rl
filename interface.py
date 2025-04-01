import torch
import numpy as np
from gridworld import Gridworld, DQN  # Assuming your environment & model code are in gridworld.py

def get_path(start, end, obstacles):
    """
    Given a start and end position (as grid coordinates, e.g., (x,y)) and a list of obstacles,
    returns a list of waypoints representing the path the drone will follow.
    
    Parameters:
      start: tuple of (x, y) for the starting grid coordinate.
      end:   tuple of (x, y) for the destination grid coordinate.
      obstacles: list of (x, y) tuples representing the positions of obstacles.
      
    Returns:
      path: a list of (x, y) waypoints from start to end.
    """
    # Create and configure the environment
    env = Gridworld()
    env.agents = [start]       # Set the agent's starting position
    env.goals = [end]          # Set the agent's goal
    env.obstacles = obstacles  # Replace obstacles with the provided list

    # Reset the environment and get the initial state for the agent
    state = env.reset()[0]  # For a single agent, take the first state

    # Determine state dimension and action dimension
    state_dim = len(state)
    action_dim = 4  # up, down, left, right

    # Load the trained model (ensure that trained_model.pth is provided)
    model = DQN(state_dim, action_dim)
    # Adjust map_location if your coworker is on CPU
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
    model.eval()

    # Run the simulation until the goal is reached (or a maximum step count)
    path = [start]
    done = False
    max_steps = 1000  # safety cutoff
    current_state = state
    while not done and len(path) < max_steps:
        # Use the trained model to choose an action (greedy selection)
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        # Step the environment (note: our step function expects a list of actions)
        next_state, reward, done = env.step([action])
        current_state = next_state[0]  # For single agent, take the first element
        # Append the new agent position to the path
        path.append(env.agents[0])
    return path

if __name__ == '__main__':
    # Example usage: convert your lat/lon to grid coordinates as needed
    start = (0, 0)
    end = (14, 14)
    # Example obstacles list (you can update this as needed)
    obstacles = [(5,5), (6,5), (7,5), (5,6)]
    path = get_path(start, end, obstacles)
    print("Computed Path (as grid waypoints):")
    print(path)
