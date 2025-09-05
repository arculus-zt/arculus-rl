# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:05:19 2025

@author: bvlsc
"""

import numpy as np

import random
random.seed(50)

import pygame
import sys
import time
import math



# Initialize Pygame
pygame.init()

# Set the dimensions of the window
width, height = 800, 800
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Truck and Points Visualization with Q-learning")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

cheese_eaten = False
vx, vy = 0, 0

# Define obstacles as rectangles
obstacles = [
    pygame.Rect(370, 300, 400, 100),
    pygame.Rect(175, 200, 100, 200),
    pygame.Rect(500 - 50, 700 - 50, 100, 100)
]

congestion = [
    pygame.Rect(75, 75, 100, 100)
]


window_size = 10



rr=8
tzr=0
fullBattery = 5000000


def save_plot_reward_to_file(plot_reward, file_name="plot_reward.txt"):
    with open(file_name, 'w') as file:
        for reward in plot_reward:
            file.write(f"{reward}\n")
    # print(f"Plot rewards saved to {file_name}")


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Generate random points within the window bounds
def generate_random_point():
    return (random.randint(100, width-100), random.randint(100, height-100))
    # return (width // 2, height // 2)

# Q-learning parameters
alpha = 0.2
gamma = 0.95
# epsilon = 0.2


episodes = 2000







# Initialize video recording for first two and last two episodes
capture_episodes = [90]





plot_reward = []

# Initialize Q-table
states = []
for x in range(0, width, 20):
    for y in range(0, height, 20):
        # for b in range(0, 1000):
            for cheese_eaten in [True, False]:
                for near_congestion in [True, False]:
                    for s in ['left', 'right', 'up', 'down', 'down_left', 'down_right', 'up_left', 'up_right', None]:
                        for near_customer in [True, False]:
                            for near_depot in [True, False]:
                                for near_emergency in [True, False]:
                                    for batteryLow in [True, False]:
                                        #print(((x, y), cheese_eaten, batteryLow, (near_congestion, s), (near_customer, near_depot, near_emergency)))
                                        states.append(((x, y), cheese_eaten, batteryLow, (near_congestion, s), (near_customer, near_depot, near_emergency)))
                                        
# print(states)
actions = [
    'move_to_customer', 'move_to_depot', 'move_to_emergency',
    'move_up', 'move_down', 'move_left', 'move_right',
    'move_up_left', 'move_up_right', 'move_down_left', 'move_down_right',
    'increase_speed', 'decrease_speed', 
    'move_away'
]
q_table = {}
for state in states:
    q_table[state] = {action: 0 for action in actions}
    
    
q_table2 = {}
for state in states:
    q_table2[state] = {action: 0 for action in actions}

def is_in_congestion_zone(drone):
    for cong in congestion:
        if cong.collidepoint(drone):
            return True
    return False

def is_near_congestion(drone):
    if not congestion:
        return False, None

    nearest_congestion = min(congestion, key=lambda cong: distance(drone, (cong.x + cong.width // 2, cong.y + cong.height // 2)))
    nearest_distance = distance(drone, (nearest_congestion.x + nearest_congestion.width // 2, nearest_congestion.y + nearest_congestion.height // 2))
    
    # Round to the nearest multiple of 20
    nearest_distance = round(nearest_distance / 20) * 20
    near = nearest_distance < 100  # Consider near if within 100 units

    # Calculate the center of the nearest congestion zone
    congestion_center_x = nearest_congestion.x + nearest_congestion.width // 2
    congestion_center_y = nearest_congestion.y + nearest_congestion.height // 2

    # Determine the primary direction of the nearest congestion zone
    if congestion_center_x < drone[0] and congestion_center_y < drone[1]:
        direction = 'up_left'
    elif congestion_center_x > drone[0] and congestion_center_y < drone[1]:
        direction = 'up_right'
    elif congestion_center_x < drone[0] and congestion_center_y > drone[1]:
        direction = 'down_left'
    elif congestion_center_x > drone[0] and congestion_center_y > drone[1]:
        direction = 'down_right'
    elif congestion_center_x < drone[0]:
        direction = 'left'
    elif congestion_center_x > drone[0]:
        direction = 'right'
    elif congestion_center_y < drone[1]:
        direction = 'up'
    else:
        direction = 'down'

    if near:
        return near, direction
    return near, None

# Function to move the drone towards a target
def move_towards(drone, target, speed=2):
    global vx, vy
    if target is None:
        vx, vy = 0, 0
        return drone

    x1, y1 = drone
    x2, y2 = target

    dx, dy = x2 - x1, y2 - y1
    distance = np.hypot(dx, dy)

    if distance < speed:
        vx, vy = 0, 0
        return target

    vx, vy = (dx / distance) * speed, (dy / distance) * speed
    x1 += vx
    y1 += vy

    return x1, y1

def distance(point1, point2):
    return np.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)

# Function to check if within radius
def is_within_radius(point1, point2, radius=20):
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1]) <= radius

# Function to get reward
def get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery):
    for obstacle in obstacles:
            if obstacle.collidepoint(drone):
                # print("collide bad")
                return -100
    if drone[0] == width - 1 or drone[0] == 0 or drone[1] == height or drone[1] == 0:
        return -100
    if battery <= 0:
        
        return -10000
    reward = 0
    
    
    
    if is_in_congestion_zone(drone):
            reward -= 50
    

    if not near_customer and distance(drone, customer) < 50:
        reward += 20
    if not near_depot and distance(drone, depot) < 50:
        reward += 20
    if not near_emergency and distance(drone, emergency) < 50:
        reward += 20

    if not delivered:
        if is_within_radius(drone, customer):
            return reward + 50*rr
        
        

    if is_within_radius(drone, depot):
        return reward + 50
   
    
    if is_within_radius(drone, emergency):
        return reward + 50
       
    
    if distance((100,100), drone) < tzr:                    
        return reward - 0.005*(abs(vx) + abs(vy))  # Penalty proportional to velocity
    else:
        return reward - 0.0005*(abs(vx) + abs(vy))

def min_max_scaling(q_values):
    min_q = min(q_values)
    max_q = max(q_values)
    if max_q == min_q:
        return [0.5 for _ in q_values]  # Return 0.5 for all elements if all Q-values are the same
    return [(q - min_q) / (max_q - min_q) for q in q_values]


def softmax(q_values):
    # print("Q", q_values)
    max_q = max(q_values)
    temperature = 0.7
    exp_values = [math.exp((q-max_q) / temperature) for q in q_values]
    # print("E", exp_values)
    total = sum(exp_values)
    probabilities = [exp / total for exp in exp_values]
    # print("P", probabilities)
    return probabilities
    
# Function to choose an action
def choose_action(state, epsilon, q_table):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        q_values = [q_table[state][action] for action in actions]
        scaled = min_max_scaling(q_values)
        probabilities = softmax(scaled)
        return random.choices(actions, probabilities)[0]


def generate_dynamic_congestion(episode):
    congestion.clear()
    num_congestion_zones = random.randint(1, 4)  # Adjust the number of zones as needed
    for _ in range(num_congestion_zones):
        x = random.randint(0, (width // 20) - 5) * 20  # Generate x-coordinate aligned to multiple of 20
        y = random.randint(0, (height // 20) - 5) * 20
        congestion.append(pygame.Rect(x, y, 100, 100))

def add_random_congestion(congestion_list, width, height):
    new_x = random.randint(0, (width // 20) - 5) * 20
    new_y = random.randint(0, (height // 20) - 5) * 20
    
    new_congestion = pygame.Rect(new_x, new_y, 100, 100)
    congestion_list.append(new_congestion)

def move_away_from_congestion(state):
    drone = state[0]
    nearest_congestion = min(congestion, key=lambda cong: distance(drone, (cong.x + cong.width // 2, cong.y + cong.height // 2)))
    #nearest_distance = distance(drone, (nearest_congestion.x + nearest_congestion.width // 2, nearest_congestion.y + nearest_congestion.height // 2))

    congestion_center_x = nearest_congestion.x + nearest_congestion.width // 2
    congestion_center_y = nearest_congestion.y + nearest_congestion.height // 2

    # Determine the primary direction of the nearest congestion zone
    direction = (0, 0)
    if congestion_center_x < drone[0] and congestion_center_y < drone[1]:
        direction = (-1, -1)
    elif congestion_center_x > drone[0] and congestion_center_y < drone[1]:
        direction = (1, -1)
    elif congestion_center_x < drone[0] and congestion_center_y > drone[1]:
        direction = (-1, 1)
    elif congestion_center_x > drone[0] and congestion_center_y > drone[1]:
        direction = (1, 1)
    elif congestion_center_x < drone[0]:
        direction = (-1, 0)
    elif congestion_center_x > drone[0]:
        direction = (1, 0)
    elif congestion_center_y < drone[1]:
       direction = (0, -1)
    else:
        direction = (0, 1)

    # print(directionOfZone)
    # Calculate the new position
    new_x = drone[0] + direction[0] * -50
    new_y = drone[1] + direction[1] * -50

    # Ensure the new position is within bounds
    new_x = max(0, min(width - 1, new_x))
    new_y = max(0, min(height - 1, new_y))

    return (new_x, new_y), direction

        


# Font for displaying text
font = pygame.font.Font(None, 36)
total_rewards = []
delivered_rewards = []
path = set()
# Training the Q-learning agent with visualization

time_congestion=0


for episode in range(episodes):
    print("Training 1: Ep.", episode)
    
    customer = (100, 700)
    
    time_congestion=0
    
    congestion.clear()
    generate_dynamic_congestion(episode)
    if episode % 20000 == 0:
        congestion.pop()
        new_congestion = pygame.Rect(280, 300, 100, 100)
        congestion.append(new_congestion)
    path = set()
    drone = (100, 100)
    truck = generate_random_point()
    
    depot = (700, 700)
    emergency = (700, 150)
    # epsilon = 0.2
    epsilon = max(0.01, 1 - episode / (0.95*episodes)) 
    delivered = False
    near_customer = False
    near_depot = False
    near_emergency = False
    target = None
    steps = 0
    viz_training = True
    viz_step=99
    total_reward = 0
    speed = 1
    state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, False, is_near_congestion(drone), (distance(drone, customer) < 50, distance(drone, depot) < 50, distance(drone, emergency) < 50))
    dir = None

    battery = fullBattery
    while not is_within_radius(drone, depot) and not is_within_radius(drone, emergency) and battery > 0:
        if steps % 200 == 0:
            if steps != 0:
                random.shuffle(congestion)
                congestion.pop()
            add_random_congestion(congestion, 800, 800)
        # epsilon *= 0.9995
        if state not in q_table:
            q_table[state] = {action: 0 for action in actions}
        action = choose_action(state, epsilon, q_table)
        dronePrev = drone
        
        if action == 'move_to_customer':
            target = customer
            drone = move_towards(drone, target, speed)
        elif action == 'move_to_depot':
            target = depot
            drone = move_towards(drone, target, speed)
        elif action == 'move_to_emergency':
            target = emergency
            drone = move_towards(drone, target, speed)
        elif action == 'move_up':
            target = (drone[0], max(drone[1] - 20, 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down':
            target = (drone[0], min(drone[1] + 20, height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'move_left':
            target = (max(drone[0] - 20, 0), drone[1])
            drone = move_towards(drone, target, speed)
        elif action == 'move_right':
            target = (min(drone[0] + 20, width-1), drone[1])
            drone = move_towards(drone, target, speed)
        elif action == 'move_up_left':
            target = (max(drone[0] - math.sqrt(20), 0), max(drone[1] - math.sqrt(20), 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_up_right':
            target = (min(drone[0] + math.sqrt(20), width-1), max(drone[1] - math.sqrt(20), 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down_left':
            target = (max(drone[0] - math.sqrt(20), 0), min(drone[1] + math.sqrt(20), height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down_right':
            target = (min(drone[0] + math.sqrt(20), width-1), min(drone[1] + math.sqrt(20), height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'increase_speed':
            if speed < 20:
                speed += 1
            else:
                speed = 20
        elif action == 'decrease_speed':
            if speed > 1:
                speed -= 1
            else:
                speed = 1
        elif action == 'move_away':

            target, dir = move_away_from_congestion(state)
            drone = move_towards(drone, target, speed)
        
        

    
        battery = round(battery - 0.5 * 1 * (speed ** 2)) if battery > 0 else 0
        isBatteryLow = False if battery / fullBattery > 0.25 else True
        # battery = round(battery / 5) * 5
        
        # print(battery)
    
        reward = get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery)

        if battery < 0:
            battery = 0

        for obstacle in obstacles:
            if obstacle.collidepoint(drone):
                drone = dronePrev
        
        path.add(drone)
        # if action == 'move_to_customer':
        #     print(reward)
        
        if is_in_congestion_zone(drone):
            time_congestion+=1
        
        if distance(drone, customer) < 50:
            near_customer = True
    
        if distance(drone, depot) < 50:
            near_depot = True

        if distance(drone, emergency) < 50:
            near_emergency = True
        
        if not delivered and is_within_radius(drone, customer):
            # print("delivered")
            delivered = True
        
    
        total_reward += reward
        next_state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, isBatteryLow, is_near_congestion(drone), (near_customer, near_depot, near_emergency))
        if next_state not in q_table:
            q_table[next_state] = {action: 0 for action in actions}
        # if cheese_eaten:
        #     print(next_state)
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[state][action])
        state = next_state

        
        steps += 1

        
        if delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
            returned = True
            break
        
        
        if not delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
            returned = True
            break
  
        
for episode in range(episodes):
    print("Training 2: Ep.", episode)
    
    customer = (700, 400)
    
    time_congestion=0
    
    congestion.clear()
    generate_dynamic_congestion(episode)
    if episode % 20000 == 0:
        congestion.pop()
        new_congestion = pygame.Rect(280, 300, 100, 100)
        congestion.append(new_congestion)
    path = set()
    drone = (100, 100)
    truck = generate_random_point()
    
    depot = (700, 700)
    emergency = (700, 150)
    # epsilon = 0.2
    epsilon = max(0.01, 1 - episode / (0.95*episodes)) 
    delivered = False
    near_customer = False
    near_depot = False
    near_emergency = False
    target = None
    steps = 0
    viz_training = True
    viz_step=99
    total_reward = 0
    speed = 1
    state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, False, is_near_congestion(drone), (distance(drone, customer) < 50, distance(drone, depot) < 50, distance(drone, emergency) < 50))
    dir = None

    battery = fullBattery
    while not is_within_radius(drone, depot) and not is_within_radius(drone, emergency) and battery > 0:
        if steps % 200 == 0:
            if steps != 0:
                random.shuffle(congestion)
                congestion.pop()
            add_random_congestion(congestion, 800, 800)
        # epsilon *= 0.9995
        if state not in q_table2:
            q_table2[state] = {action: 0 for action in actions}
        action = choose_action(state, epsilon, q_table2)
        dronePrev = drone
        
        if action == 'move_to_customer':
            target = customer
            drone = move_towards(drone, target, speed)
        elif action == 'move_to_depot':
            target = depot
            drone = move_towards(drone, target, speed)
        elif action == 'move_to_emergency':
            target = emergency
            drone = move_towards(drone, target, speed)
        elif action == 'move_up':
            target = (drone[0], max(drone[1] - 20, 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down':
            target = (drone[0], min(drone[1] + 20, height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'move_left':
            target = (max(drone[0] - 20, 0), drone[1])
            drone = move_towards(drone, target, speed)
        elif action == 'move_right':
            target = (min(drone[0] + 20, width-1), drone[1])
            drone = move_towards(drone, target, speed)
        elif action == 'move_up_left':
            target = (max(drone[0] - math.sqrt(20), 0), max(drone[1] - math.sqrt(20), 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_up_right':
            target = (min(drone[0] + math.sqrt(20), width-1), max(drone[1] - math.sqrt(20), 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down_left':
            target = (max(drone[0] - math.sqrt(20), 0), min(drone[1] + math.sqrt(20), height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down_right':
            target = (min(drone[0] + math.sqrt(20), width-1), min(drone[1] + math.sqrt(20), height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'increase_speed':
            if speed < 20:
                speed += 1
            else:
                speed = 20
        elif action == 'decrease_speed':
            if speed > 1:
                speed -= 1
            else:
                speed = 1
        elif action == 'move_away':

            target, dir = move_away_from_congestion(state)
            drone = move_towards(drone, target, speed)
        
        

    
        battery = round(battery - 0.5 * 1 * (speed ** 2)) if battery > 0 else 0
        isBatteryLow = False if battery / fullBattery > 0.25 else True
        # battery = round(battery / 5) * 5
        
        # print(battery)
    
        reward = get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery)

        if battery < 0:
            battery = 0

        for obstacle in obstacles:
            if obstacle.collidepoint(drone):
                drone = dronePrev
        
        path.add(drone)
        # if action == 'move_to_customer':
        #     print(reward)
        
        if is_in_congestion_zone(drone):
            time_congestion+=1
        
        if distance(drone, customer) < 50:
            near_customer = True
    
        if distance(drone, depot) < 50:
            near_depot = True

        if distance(drone, emergency) < 50:
            near_emergency = True
        
        if not delivered and is_within_radius(drone, customer):
            # print("delivered")
            delivered = True
        
    
        total_reward += reward
        next_state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, isBatteryLow, is_near_congestion(drone), (near_customer, near_depot, near_emergency))
        if next_state not in q_table2:
            q_table2[next_state] = {action: 0 for action in actions}
        # if cheese_eaten:
        #     print(next_state)
        q_table2[state][action] = q_table2[state][action] + alpha * (reward + gamma * max(q_table2[next_state].values()) - q_table2[state][action])
        state = next_state

        
        steps += 1
        
        
        if delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
            returned = True
            break
        
        
        if not delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
            returned = True
            break




epsilon = 0.95
def choose_viz_action(state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        # Exploit: Choose best action
        return max(q_table[state], key=q_table[state].get)
    else:
        # Explore: Choose random action
        return random.choice(list(q_table[state].keys()))




print("Starting Visualization Mode...")

testing_episodes = 10  # Number of visualization episodes

for episode in range(testing_episodes):
    print(f"Visualization Episode {episode + 1}")

    congestion.clear()
    generate_dynamic_congestion(episode)

    if episode % 20000 == 0:
        congestion.pop()
        new_congestion = pygame.Rect(280, 300, 100, 100)
        congestion.append(new_congestion)

    # Initialize drones
    drone1 = (100, 100)
    drone2 = (100, 100)
    truck = generate_random_point()
    customer1 = (100, 700)
    customer2 = (700, 400)
    depot = (700, 700)
    emergency = (700, 150)

    delivered1 = False
    delivered2 = False
    speed = 5
    battery1 = fullBattery
    battery2 = fullBattery
    path1 = set()
    path2 = set()
    
    steps = 0
    running = True

    while running:
        # Event handling (allow closing window manually)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update congestion dynamically
        if steps % 200 == 0 and steps != 0:
            random.shuffle(congestion)
            congestion.pop()
            add_random_congestion(congestion, 800, 800)

        # Get state for both drones
        state1 = ((int(drone1[0] // 20) * 20, int(drone1[1] // 20) * 20), delivered1,
                  battery1 / fullBattery < 0.25, is_near_congestion(drone1),
                  (distance(drone1, customer1) < 50, distance(drone1, depot) < 50, distance(drone1, emergency) < 50))

        state2 = ((int(drone2[0] // 20) * 20, int(drone2[1] // 20) * 20), delivered2,
                  battery2 / fullBattery < 0.25, is_near_congestion(drone2),
                  (distance(drone2, customer2) < 50, distance(drone2, depot) < 50, distance(drone2, emergency) < 50))

        # Choose best action from Q-tables (no exploration, only exploitation)
        action1 = choose_viz_action(state1, q_table, epsilon)
        action2 = choose_viz_action(state2, q_table2, epsilon)
        
        # Move Drone 1 based on action1
        if action1 == 'move_to_customer':
            drone1 = move_towards(drone1, customer1, speed)
        elif action1 == 'move_to_depot':
            drone1 = move_towards(drone1, depot, speed)
        elif action1 == 'move_to_emergency':
            drone1 = move_towards(drone1, emergency, speed)
        elif action1 == 'move_up':
            drone1 = move_towards(drone1, (drone1[0], max(drone1[1] - 20, 0)), speed)
        elif action1 == 'move_down':
            drone1 = move_towards(drone1, (drone1[0], min(drone1[1] + 20, height-1)), speed)
        elif action1 == 'move_left':
            drone1 = move_towards(drone1, (max(drone1[0] - 20, 0), drone1[1]), speed)
        elif action1 == 'move_right':
            drone1 = move_towards(drone1, (min(drone1[0] + 20, width-1), drone1[1]), speed)
        elif action1 == 'move_up_left':
            drone1 = move_towards(drone1, (max(drone1[0] - math.sqrt(20), 0), max(drone1[1] - math.sqrt(20), 0)), speed)
        elif action1 == 'move_up_right':
            drone1 = move_towards(drone1, (min(drone1[0] + math.sqrt(20), width-1), max(drone1[1] - math.sqrt(20), 0)), speed)
        elif action1 == 'move_down_left':
            drone1 = move_towards(drone1, (max(drone1[0] - math.sqrt(20), 0), min(drone1[1] + math.sqrt(20), height-1)), speed)
        elif action1 == 'move_down_right':
            drone1 = move_towards(drone1, (min(drone1[0] + math.sqrt(20), width-1), min(drone1[1] + math.sqrt(20), height-1)), speed)
        elif action1 == 'move_away':
            target1, _ = move_away_from_congestion(state1)
            drone1 = move_towards(drone1, target1, speed)
        
        # Move Drone 2 based on action2
        if action2 == 'move_to_customer':
            drone2 = move_towards(drone2, customer2, speed)
        elif action2 == 'move_to_depot':
            drone2 = move_towards(drone2, depot, speed)
        elif action2 == 'move_to_emergency':
            drone2 = move_towards(drone2, emergency, speed)
        elif action2 == 'move_up':
            drone2 = move_towards(drone2, (drone2[0], max(drone2[1] - 20, 0)), speed)
        elif action2 == 'move_down':
            drone2 = move_towards(drone2, (drone2[0], min(drone2[1] + 20, height-1)), speed)
        elif action2 == 'move_left':
            drone2 = move_towards(drone2, (max(drone2[0] - 20, 0), drone2[1]), speed)
        elif action2 == 'move_right':
            drone2 = move_towards(drone2, (min(drone2[0] + 20, width-1), drone2[1]), speed)
        elif action2 == 'move_up_left':
            drone2 = move_towards(drone2, (max(drone2[0] - math.sqrt(20), 0), max(drone2[1] - math.sqrt(20), 0)), speed)
        elif action2 == 'move_up_right':
            drone2 = move_towards(drone2, (min(drone2[0] + math.sqrt(20), width-1), max(drone2[1] - math.sqrt(20), 0)), speed)
        elif action2 == 'move_down_left':
            drone2 = move_towards(drone2, (max(drone2[0] - math.sqrt(20), 0), min(drone2[1] + math.sqrt(20), height-1)), speed)
        elif action2 == 'move_down_right':
            drone2 = move_towards(drone2, (min(drone2[0] + math.sqrt(20), width-1), min(drone2[1] + math.sqrt(20), height-1)), speed)
        elif action2 == 'move_away':
            target2, _ = move_away_from_congestion(state2)
            drone2 = move_towards(drone2, target2, speed)


        # Check if drones reached customers
        if is_within_radius(drone1, customer1):
            delivered1 = True
        if is_within_radius(drone2, customer2):
            delivered2 = True

        # Reduce battery (but not too quickly)
        battery1 = max(battery1 - 5, 0)
        battery2 = max(battery2 - 5, 0)

        # Collision handling
        for obstacle in obstacles:
            if obstacle.collidepoint(drone1):
                drone1 = (100, 100)  # Reset drone position
            if obstacle.collidepoint(drone2):
                drone2 = (100, 100)

        # Update paths
        path1.add(drone1)
        path2.add(drone2)

        # Draw environment
        window.fill(WHITE)
        pygame.draw.circle(window, RED, drone1, 5)
        pygame.draw.circle(window, BLUE, drone2, 5)

        if not delivered1:
            pygame.draw.circle(window, GREEN, customer1, 20)
        if not delivered2:
            pygame.draw.circle(window, GREEN, customer2, 20)

        pygame.draw.circle(window, BLUE, depot, 20)
        pygame.draw.circle(window, CYAN, emergency, 20)

        for cong in congestion:
            pygame.draw.rect(window, RED, cong)
        for obstacle in obstacles:
            pygame.draw.rect(window, ORANGE, obstacle)

        # Draw paths
        for p in path1:
            pygame.draw.circle(window, (0, 255, 0), p, 3)  # Green path for Drone 1
        for p in path2:
            pygame.draw.circle(window, (0, 0, 255), p, 3)  # Blue path for Drone 2

        pygame.display.flip()
        time.sleep(0.05)  # Slow down for visualization

        # Stopping Condition (when both drones finish their deliveries)
        if (is_within_radius(drone1, depot) or is_within_radius(drone1, emergency)) and (is_within_radius(drone2, depot) or is_within_radius(drone2, emergency)):
            running = False

        steps += 1

print("Visualization Complete.")
pygame.quit()
sys.exit()




