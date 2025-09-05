# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:09:05 2024

@author: bvlsc
"""

import numpy as np
import random
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
# Generate random points within the window bounds
def generate_random_point():
    # random.seed(128)
    return (random.randint(100, width-100), random.randint(100, height-100))
    # return (width // 2, height // 2)

def generate_congestion_zone():
    return pygame.Rect(random.randint(0, width-100), random.randint(0, height-100), 100, 100)

# Q-learning parameters
alpha = 0.2
gamma = 0.95
# epsilon = 0.2
episodes = 100001

# Initialize Q-table
states = []
for x in range(0, width, 20):
    for y in range(0, width, 20):
        for cheese_eaten in [True, False]:
            states.append(((x, y), cheese_eaten))
print(states)
actions = [
    'move_to_customer', 'move_to_depot', 'move_to_emergency',
    'move_up', 'move_down', 'move_left', 'move_right',
    'move_up_left', 'move_up_right', 'move_down_left', 'move_down_right',
    'increase_speed', 'decrease_speed'
]
q_table = {}
for state in states:
    q_table[state] = {action: 0 for action in actions}

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
def get_reward(drone, customer, emergency, depot, delivered, truck, speed, congestion_zone):
    for obstacle in obstacles:
        if obstacle.collidepoint(drone):
            return -1000000000000
    if drone[0] == width - 1 or drone[0] == 0 or drone[1] == height or drone[1] == 0:
        return -1000000000000
    reward = 0
    if congestion_zone and congestion_zone.collidepoint(drone):
        reward -= 1000

    reward -= (distance(drone, truck) / 100 - 0.1 * (abs(vx) + abs(vy)))

    if not delivered:
        if is_within_radius(drone, customer):
            return reward + 5000
    else:
        if is_within_radius(drone, depot):
            return reward + 25
        if is_within_radius(drone, emergency):
            return reward + 5000

    return reward - 10 - 0.1 * (abs(vx) + abs(vy))  # Penalty proportional to velocity
    
# Function to choose an action
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return max(q_table[state], key=q_table[state].get)

# Font for displaying text
font = pygame.font.Font(None, 36)
total_rewards = []
delivered_rewards = []
path = set()
# Training the Q-learning agent with visualization
for episode in range(episodes):
    print(episode)
    path = set()
    drone = (100, 100)
    truck = generate_random_point()
    customer = (100, 700)
    depot = (700, 700)
    emergency = (700, 150)
    congestion_zone = None
    congestion_timer = 0
    epsilon = max(0.01, 0.9 - episode / 50000) 
    delivered = False
    target = None
    steps = 0
    total_reward = 0
    speed = 1
    state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered)
    viz_step = 500

    while not is_within_radius(drone, depot) and not is_within_radius(drone, emergency):
        # Handle congestion zone appearance and disappearance
        if congestion_timer == 0:
            if random.random() < 0.5:  # 50% chance to create or remove congestion zone
                if congestion_zone is None:
                    congestion_zone = generate_congestion_zone()
                else:
                    congestion_zone = None
            congestion_timer = random.randint(50, 200)  # Random duration before next change
        else:
            congestion_timer -= 1

        action = choose_action(state, epsilon)
        dronePrev = drone
        for obstacle in obstacles:
            if obstacle.collidepoint(drone):
                drone = dronePrev
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
            target = (drone[0], max(drone[1] - 20 * speed, 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down':
            target = (drone[0], min(drone[1] + 20 * speed, height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'move_left':
            target = (max(drone[0] - 20 * speed, 0), drone[1])
            drone = move_towards(drone, target, speed)
        elif action == 'move_right':
            target = (min(drone[0] + 20 * speed, width-1), drone[1])
            drone = move_towards(drone, target, speed)
        elif action == 'move_up_left':
            target = (max(drone[0] - math.sqrt(20 * speed), 0), max(drone[1] - math.sqrt(20 * speed), 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_up_right':
            target = (min(drone[0] + math.sqrt(20 * speed), width-1), max(drone[1] - math.sqrt(20 * speed), 0))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down_left':
            target = (max(drone[0] - math.sqrt(20 * speed), 0), min(drone[1] + math.sqrt(20 * speed), height-1))
            drone = move_towards(drone, target, speed)
        elif action == 'move_down_right':
            target = (min(drone[0] + math.sqrt(20 * speed), width-1), min(drone[1] + math.sqrt(20 * speed), height-1))
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

        path.add(drone)

        reward = get_reward(drone, customer, emergency, depot, delivered, truck, speed, congestion_zone)
        if not delivered and is_within_radius(drone, customer):
            delivered = True

        total_reward += reward
        next_state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered)
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[state][action])
        state = next_state

        if episode % viz_step == 0 and episode != 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            window.fill(WHITE)
            pygame.draw.circle(window, RED, drone, 5)
            if not delivered:
                pygame.draw.circle(window, GREEN, customer, 20)
            pygame.draw.circle(window, BLUE, depot, 20)
            pygame.draw.circle(window, CYAN, emergency, 20)
            if congestion_zone:
                pygame.draw.rect(window, YELLOW, congestion_zone)
            for obstacle in obstacles:
                pygame.draw.rect(window, BLACK, obstacle)
            for p in path:
                pygame.draw.circle(window, (0, 255, 0), p, 5)

            if target is not None:
                pygame.draw.line(window, GREEN, drone, customer, 1)
                pygame.draw.line(window, BLUE, drone, depot, 1)
                pygame.draw.line(window, CYAN, drone, emergency, 1)
            pygame.draw.circle(window, BLACK, truck, 5)

            vx_text = font.render(f'Episode: {episode + 1}', True, BLACK)
            vy_text = font.render(f'Reward: {reward:.2f}', True, BLACK)
            window.blit(vx_text, (10, 10))
            window.blit(vy_text, (10, 50))

            pygame.display.flip()
            time.sleep(0.01)

        steps += 1
    if delivered:
        delivered_rewards.append(total_reward)
    else:
        total_rewards.append(total_reward)


print("final path")
for p in path:
    print(p)

pygame.quit()
sys.exit()
