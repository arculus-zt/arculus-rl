# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:05:19 2025

@author: bvlsc
"""

import numpy as np

import random
random.seed(82)

import pygame
import sys
import time
import math
import os

import imageio

# Directory to save videos
output_dir = "videos/"
os.makedirs(output_dir, exist_ok=True)



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


episodes = 10000







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
    return np.linalg.norm(np.array(point1) - np.array(point2))

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


testing=10
video_writers = [
    imageio.get_writer(f"{output_dir}/episode_{episode+1}.mp4", fps=30)
    for episode in range(testing)
]



for episode in range(testing):
    # print("Test Ep.", episode)
    video_writer = video_writers[episode]

    congestion.clear()
    generate_dynamic_congestion(episode)
    if episode % 20000 == 0:
        congestion.pop()
        new_congestion = pygame.Rect(280, 300, 100, 100)
        congestion.append(new_congestion)
    
    
    path = set()
    path2= set()
    
    
    drone = (100, 100)
    drone2 = (100,100)
    
    truck = generate_random_point()
    
    customer = (100, 700)
    customer2 = (700, 400)
    
    depot = (700, 700)
    emergency = (700, 150)
    
    delivered = False
    delivered2 = False
    
    
    near_customer = False
    near_depot = False
    near_emergency = False
    near_customer2 = False
    near_depot2 = False
    near_emergency2 = False
    
    
    target = None
    
    returned = False
    returned2 = False
    
    steps = 0
    steps2 = 0
    


    total_reward = 0
    
    speed = 1
    speed2 = 1
    
    state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, False, is_near_congestion(drone), (distance(drone, customer) < 50, distance(drone, depot) < 50, distance(drone, emergency) < 50))
    state2 = ((int(drone2[0] // 20) * 20, int(drone2[1] // 20) * 20), delivered2, False, is_near_congestion(drone2), (distance(drone2, customer2) < 50, distance(drone2, depot) < 50, distance(drone2, emergency) < 50))
    
    dir = None
    
    battery = fullBattery
    battery2 = fullBattery
    
    
    while all(not is_within_radius(drone, loc) for loc in [depot, emergency]) and \
      all(not is_within_radius(drone2, loc) for loc in [depot, emergency]) and \
      battery > 0 and battery2 > 0:

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
        
        
        
        
        if state2 not in q_table2:
            q_table2[state2] = {action: 0 for action in actions}
        action2 = choose_action(state2, epsilon, q_table2)
        dronePrev2 = drone2
        
        if action2 == 'move_to_customer':
            target = customer2
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_to_depot':
            target = depot
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_to_emergency':
            target = emergency
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_up':
            target = (drone2[0], max(drone2[1] - 20, 0))
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_down':
            target = (drone2[0], min(drone2[1] + 20, height-1))
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_left':
            target = (max(drone2[0] - 20, 0), drone2[1])
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_right':
            target = (min(drone2[0] + 20, width-1), drone2[1])
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_up_left':
            target = (max(drone2[0] - math.sqrt(20), 0), max(drone2[1] - math.sqrt(20), 0))
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_up_right':
            target = (min(drone2[0] + math.sqrt(20), width-1), max(drone2[1] - math.sqrt(20), 0))
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_down_left':
            target = (max(drone2[0] - math.sqrt(20), 0), min(drone2[1] + math.sqrt(20), height-1))
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'move_down_right':
            target = (min(drone2[0] + math.sqrt(20), width-1), min(drone2[1] + math.sqrt(20), height-1))
            drone2 = move_towards(drone2, target, speed2)
        elif action2 == 'increase_speed':
            if speed2 < 20:
                speed2 += 1
            else:
                speed2 = 20
        elif action2 == 'decrease_speed':
            if speed2 > 1:
                speed2 -= 1
            else:
                speed2 = 1
        elif action2 == 'move_away':
    
            target, dir = move_away_from_congestion(state2)
            drone2 = move_towards(drone2, target, speed2)
        
        
        
        
        # Update delivery status if drone reaches customer
    
    
        battery = round(battery - 0.5 * 1 * (speed ** 2)) if battery > 0 else 0
        battery2 = round(battery2 - 0.5 * 1 * (speed2 ** 2)) if battery2 > 0 else 0
        
        isBatteryLow = False if battery / fullBattery > 0.25 else True
        isBatteryLow2 = False if battery2 / fullBattery > 0.25 else True
        # battery = round(battery / 5) * 5
        
        # print(battery)
    
        reward = get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery)
        reward2 = get_reward(drone2, customer, emergency, depot, delivered2, truck, speed2, near_customer, near_depot, near_emergency, battery2)
    
        if battery < 0:
            battery = 0
            
        if battery2 < 0:
            battery2 = 0

                
                
    
        for obstacle in obstacles:
            if obstacle.collidepoint(drone):
                drone = dronePrev
                
            if obstacle.collidepoint(drone2):
                drone2 = dronePrev2
        
        
        path.add(drone)
        path2.add(drone2)
        
        # if action == 'move_to_customer':
        #     print(reward)
        
        if distance(drone, customer) < 50:
            near_customer = True
    
        if distance(drone, depot) < 50:
            near_depot = True
    
        if distance(drone, emergency) < 50:
            near_emergency = True
            
            
        if distance(drone2, customer2) < 50:
            near_customer2 = True
    
        if distance(drone2, depot) < 50:
            near_depot2 = True
    
        if distance(drone2, emergency) < 50:
            near_emergency2 = True    
            
        
        
        if not delivered and is_within_radius(drone, customer):
            # print("delivered")
            delivered = True
            
        if not delivered2 and is_within_radius(drone2, customer2):
            # print("delivered")
            delivered2 = True
        
    
        total_reward = total_reward + reward + reward2
        
        next_state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, isBatteryLow, is_near_congestion(drone), (near_customer, near_depot, near_emergency))
        next_state2 = ((int(drone2[0] // 20) * 20, int(drone2[1] // 20) * 20), delivered2, isBatteryLow2, is_near_congestion(drone2), (near_customer2, near_depot2, near_emergency2))
        
        if next_state not in q_table:
            q_table[next_state] = {action: 0 for action in actions}
            
        if next_state2 not in q_table2:
            q_table2[next_state2] = {action: 0 for action in actions}
        # if cheese_eaten:
        #     print(next_state)
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[state][action])
        state = next_state
        
        q_table2[state2][action2] = q_table2[state2][action2] + alpha * (reward + gamma * max(q_table2[next_state2].values()) - q_table2[state2][action2])
        state2 = next_state2
        
        
        # Visualization

            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        window.fill(WHITE)

        pygame.draw.circle(window, RED, drone, 5)
        pygame.draw.circle(window, BLUE, drone2, 5)
        
        
        if not delivered:
            pygame.draw.circle(window, GREEN, customer, 20)
            
        if not delivered2:
            pygame.draw.circle(window, GREEN, customer2, 20)
            
            
        pygame.draw.circle(window, BLUE, depot, 20)
        pygame.draw.circle(window, CYAN, emergency, 20)
        
        for cong in congestion:
            pygame.draw.rect(window, RED, cong)
        for obstacle in obstacles:
            pygame.draw.rect(window, ORANGE, obstacle)
        for p in path:
            pygame.draw.circle(window, (0, 255, 0), p, 5)
            
        for p in path2:
            pygame.draw.circle(window, (0, 255, 255), p, 5)
            
        # Draw obstacles
        

       

        if target is not None:
            pygame.draw.line(window, GREEN, drone, customer, 1)
            pygame.draw.line(window, BLUE, drone, depot, 1)
            pygame.draw.line(window, CYAN, drone, emergency, 1)
            
        pygame.draw.circle(window, BLACK, truck, 5)

        vx_text = font.render(f'Episode: {episode + 1}', True, BLACK)

        vy_text = font.render(f'Reward: {reward:.2f}', True, BLACK)
        batter_text = font.render(f'Battery: {battery}', True, BLACK)
        window.blit(vx_text, (10, 10))
        window.blit(vy_text, (10, 50))
        window.blit(batter_text, (10, 100))
        
        
        
        
        
        
        frame = pygame.surfarray.array3d(window)  # Convert Pygame window to array
        frame = np.rot90(frame)  # Rotate for correct orientation
        frame = np.flipud(frame)  # Flip vertically
        video_writer.append_data(frame)
        
        
        pygame.display.flip()
        time.sleep(0.01)
        
        
        
        if (delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency))):
            
            print(distance(drone, customer), distance(drone2, customer2))
            returned = True
            break
        

        
        
        steps += 1
        
        
    video_writer.close()














print("Visualization Complete.")
pygame.quit()
sys.exit()




