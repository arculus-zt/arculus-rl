# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:09:05 2024

@author: bvlsc
"""


import numpy as np
import random
random.seed(69)
import math
import time


start = time.time()


distance_cache={}

def cached_distance(point1, point2):
    # Create a key for the pair of points (order doesn't matter)
    key = tuple(sorted([point1, point2]))
    
    # If the distance is already calculated, return it
    if key in distance_cache:
        return distance_cache[key]
    
    # Otherwise, calculate the distance, store it in the cache, and return it
    dist = np.hypot(point2[0] - point1[0], point2[1] - point1[1])
    distance_cache[key] = dist
    return dist



# rewards_ratio = [3,4,5,6]
rewards_ratio = [1]
# trust_zone_radii = [0, 200, 400]
fb = [50000]
tzr=0


for fullBattery in fb:
    for rr in rewards_ratio:
    
    

        
        # Set the dimensions of the window
        width, height = 800, 800

        
        cheese_eaten = False
        vx, vy = 0, 0
        
        # Define obstacles as rectangles
        obstacles = [(370, 300, 400, 100),(175, 200, 100, 200),(500 - 50, 700 - 50, 100, 100)]
        
        congestion = [(75, 75, 100, 100)]
        
        
        window_size = 10
        
      
        # Generate random points within the window bounds
        def generate_random_point():
            return (random.randint(100, width-100), random.randint(100, height-100))
            # return (width // 2, height // 2)
        
        # Q-learning parameters
        alpha = 0.2
        gamma = 0.95
        # epsilon = 0.2
       
        episodes = 10000
        testing = 10000
    
       
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
        
        def is_in_congestion_zone(drone):
            for cong in congestion:
                # Check if the drone is within the bounds of the congestion zone
                if (cong[0] <= drone[0] <= cong[0] + cong[2]) and (cong[1] <= drone[1] <= cong[1] + cong[3]):
                    return True
            return False
        
        def is_near_congestion(drone):
            if not congestion:
                return False, None
        
            nearest_congestion = min(congestion, key=lambda cong: cached_distance(drone, (cong[0] + cong[2] // 2, cong[1] + cong[3] // 2)))
            nearest_distance = cached_distance(drone, (nearest_congestion[0] + nearest_congestion[2] // 2, nearest_congestion[1] + nearest_congestion[3] // 2))
            
            # Round to the nearest multiple of 20
            nearest_distance = round(nearest_distance / 20) * 20
            near = nearest_distance < 100  # Consider near if within 100 units
        
            # Calculate the center of the nearest congestion zone
            congestion_center_x = nearest_congestion[0] + nearest_congestion[2] // 2
            congestion_center_y = nearest_congestion[1] + nearest_congestion[3] // 2
        
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
        
            # Calculate the distance using cached_distance
            dist = cached_distance(drone, target)
        
            if dist < speed:
                vx, vy = 0, 0
                return target
        
            x1, y1 = drone
            x2, y2 = target
            dx, dy = x2 - x1, y2 - y1
        
            vx, vy = (dx / dist) * speed, (dy / dist) * speed
            x1 += vx
            y1 += vy
        
            return x1, y1

        

        
        # Function to check if within radius
        def is_within_radius(point1, point2, radius=20):
            return np.hypot(point1[0] - point2[0], point1[1] - point2[1]) <= radius
        
        # Function to get reward
        def get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery):
            for obstacle in obstacles:
                # Check if the drone is within the bounds of the obstacle
                if (obstacle[0] <= drone[0] <= obstacle[0] + obstacle[2]) and (obstacle[1] <= drone[1] <= obstacle[1] + obstacle[3]):
                    return -100  # Collision penalty
        
            if drone[0] == width - 1 or drone[0] == 0 or drone[1] == height or drone[1] == 0:
                return -100  # Penalty for going out of bounds
        
            if battery <= 0:
                return -10000  # Penalty for running out of battery
        
            reward = 0
        
            # Reward for being in the trust zone (optional feature)
            if cached_distance((100, 100), drone) < tzr:
                reward += 0.00025 * (abs(vx) + abs(vy))
        
            # Penalty for being in a congestion zone
            if is_in_congestion_zone(drone):
                reward -= 50
        
            # Rewards for being near customer, depot, or emergency
            if not near_customer and cached_distance(drone, customer) < 50:
                reward += 20
            if not near_depot and cached_distance(drone, depot) < 50:
                reward += 20
            if not near_emergency and cached_distance(drone, emergency) < 50:
                reward += 20
        
            # Reward for successful delivery
            if not delivered:
                if is_within_radius(drone, customer):
                    return reward + 50 * rr
        
            # Rewards for reaching depot or emergency after delivery
            if is_within_radius(drone, depot):
                return reward + 50
        
            if is_within_radius(drone, emergency):
                return reward + 50
        
            # Small penalty for high speed (to encourage efficient movement)
            return reward - 0.0005 * (abs(vx) + abs(vy))
        
        
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
        def choose_action(state, epsilon):
            if random.uniform(0, 1) < epsilon:
                return random.choice(actions)
            else:
                q_values = [q_table[state][action] for action in actions]
                # print("Before scale", q_values)
                scaled = min_max_scaling(q_values)
                probabilities = softmax(scaled)
                # probabilities = softmax(q_values)
                return random.choices(actions, probabilities)[0]
                # return max(q_table[state], key=q_table[state].get)
        
        def generate_dynamic_congestion(episode):
            congestion.clear()
            num_congestion_zones = random.randint(1, 4)  # Adjust the number of zones as needed
            for _ in range(num_congestion_zones):
                x = random.randint(0, (width // 20) - 5) * 20  # Generate x-coordinate aligned to multiple of 20
                y = random.randint(0, (height // 20) - 5) * 20
                congestion.append((x, y, 100, 100))
        
        def add_random_congestion(congestion_list, width, height):
            new_x = random.randint(0, (width // 20) - 5) * 20
            new_y = random.randint(0, (height // 20) - 5) * 20
            
            new_congestion = (new_x, new_y, 100, 100)
            congestion_list.append(new_congestion)
        
        def move_away_from_congestion(state):
            drone = state[0]
            nearest_congestion = min(congestion, key=lambda cong: cached_distance(drone, (cong[0] + cong[2] // 2, cong[1] + cong[3] // 2)))
            #nearest_cached_distance = distance(drone, (nearest_congestion[0] + nearest_congestion[2] // 2, nearest_congestion[1] + nearest_congestion[3] // 2))
        
            congestion_center_x = nearest_congestion[0] + nearest_congestion[2] // 2
            congestion_center_y = nearest_congestion[1] + nearest_congestion[3] // 2
        
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
        

        total_rewards = []
        delivered_rewards = []
        path = set()
        # Training the Q-learning agent with visualization
        
        time_congestion=0
        for episode in range(episodes):
            # print("Tr. Ep.", episode)
            time_congestion=0
            
            congestion.clear()
            generate_dynamic_congestion(episode)
            
            if episode % 20000 == 0:
                congestion.pop()
                new_congestion = (280, 300, 100, 100)
                congestion.append(new_congestion)
                
            path = set()
            drone = (100, 100)
            truck = generate_random_point()
            customer = (100, 700)
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
            state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, False, is_near_congestion(drone), (cached_distance(drone, customer) < 50, cached_distance(drone, depot) < 50, cached_distance(drone, emergency) < 50))
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
                action = choose_action(state, epsilon)
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
                    # Check if the drone is within the bounds of the obstacle
                    if (obstacle[0] <= drone[0] <= obstacle[0] + obstacle[2]) and (obstacle[1] <= drone[1] <= obstacle[1] + obstacle[3]):
                        drone = dronePrev
                        break
                
                path.add(drone)
                # if action == 'move_to_customer':
                #     print(reward)
                
                if is_in_congestion_zone(drone):
                    time_congestion+=1
                
                if cached_distance(drone, customer) < 50:
                    near_customer = True
            
                if cached_distance(drone, depot) < 50:
                    near_depot = True
        
                if cached_distance(drone, emergency) < 50:
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
                
    
                
            
        # print("Training Done")
        
        testing_rewards = []
        epsilon = 0
        
        
        successful_delivery_and_return = 0
        returned_without_delivery = 0
        delivery_failed_to_return = 0
        failed_deliver_failed_return = 0
        
        
        for episode in range(testing):
            # print("Test Ep.", episode)
            

        
            congestion.clear()
            generate_dynamic_congestion(episode)
            if episode % 20000 == 0:
                congestion.pop()
                new_congestion = (280, 300, 100, 100)
                congestion.append(new_congestion)
            path = set()
            drone = (100, 100)
            truck = generate_random_point()
            customer = (100, 700)
            depot = (700, 700)
            emergency = (700, 150) 
            delivered = False
            near_customer = False
            near_depot = False
            near_emergency = False
            target = None
            returned = False
            steps = 0
            viz_training = True
            viz_step=99
            total_reward = 0
            speed = 1
            state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, False, is_near_congestion(drone), (cached_distance(drone, customer) < 50, cached_distance(drone, depot) < 50, cached_distance(drone, emergency) < 50))
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
                action = choose_action(state, epsilon)
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
                
                
                
                # Update delivery status if drone reaches customer
            
            
                battery = round(battery - 0.5 * 1 * (speed ** 2)) if battery > 0 else 0
                isBatteryLow = False if battery / fullBattery > 0.25 else True
                # battery = round(battery / 5) * 5
                
                # print(battery)
            
                reward = get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery)
            
                if battery < 0:
                    battery = 0
    
                        
                        
            
                for obstacle in obstacles:
                    # Check if the drone is within the bounds of the obstacle
                    if (obstacle[0] <= drone[0] <= obstacle[0] + obstacle[2]) and (obstacle[1] <= drone[1] <= obstacle[1] + obstacle[3]):
                        drone = dronePrev
                        break
                
                path.add(drone)
                # if action == 'move_to_customer':
                #     print(reward)
                
                if cached_distance(drone, customer) < 50:
                    near_customer = True
            
                if cached_distance(drone, depot) < 50:
                    near_depot = True
            
                if cached_distance(drone, emergency) < 50:
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
                
                
                
                if delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
                    successful_delivery_and_return += 1
                    returned = True
                    break
                
                
                if not delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
                    returned_without_delivery += 1
                    returned = True
                    break
        
                
                
                
                
                
                steps += 1
                
                
            if delivered and not returned:
                delivery_failed_to_return += 1
                
            if not delivered and not returned:
                failed_deliver_failed_return += 1
                
                
            if delivered:
                delivered_rewards.append(total_reward)
            else:
                total_rewards.append(total_reward)
            
            # print(total_reward)
            
            plot_reward.append(total_reward)
        
        
        
    
        
        print("\n")   
        print("Reward Ratio: ", rr, "\tBattery: ", fullBattery)
        print(f"Successful deliveries and returns: {successful_delivery_and_return}")
        print(f"Returned without delivery: {returned_without_delivery}")
        print(f"Delivers but failed to return: {delivery_failed_to_return}")
        print(f"Both failed: {failed_deliver_failed_return}")
        print("\n")



print(time.time()-start)