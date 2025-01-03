import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt, cos, sin, pi
from collections import deque
import matplotlib.pyplot as plt
from Sharedutils import *
from collections import deque

# RL Parameters
NUM_ROBOTS = 4
STATE_DIM = 4 + (NUM_ROBOTS - 1) * 2  # Adjusted for additional robot distances
ACTION_SIZE = 5
GAMMA = 0.9
LR = 0.0003
BATCH_SIZE = 256

# Action Map
ACTION_MAP = {
    0: (0, MAX_SPEED),    # Up
    1: (0, -MAX_SPEED),   # Down
    2: (-MAX_SPEED, 0),   # Left
    3: (MAX_SPEED, 0),    # Right
    4: (0,0)              # Stay 
}

class Agent:
    def __init__(self, state_dim, action_size):
        self.policy_network = SimpleNN(STATE_DIM, ACTION_SIZE)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)
        self.memory = deque(maxlen=5000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(ACTION_SIZE)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # Compute targets
        with torch.no_grad():
            next_q_values = self.policy_network(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q_values * (1 - dones)

        q_values = self.policy_network(states)
        predictions = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = nn.functional.mse_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1)
        self.optimizer.step()

def normalize_state(state):
    return [s / GRID_SIZE for s in state]

def compute_reward(distance, prev_distance, robot_idx, robots, other_robot_distances):
    """
    Compute the reward for a given robot based on distance to target, relative distance 
    to other robots, and collisions or proximity to other robots.
    """
    # Base reward: closer distance to the target is better
    #reward = -distance / GRID_SIZE
    # Convert distance to a normalized value
    dist_norm = distance / GRID_SIZE

    reward = -dist_norm  # Negative reward proportional to normalized distance

    # Reward for improvement
    for pd in prev_distance:
        pd_norm = pd / GRID_SIZE
        if dist_norm < pd_norm:
            #reward += 0.5 * (pd_norm - dist_norm)
            reward += 1*dist_norm
            # or simply reward += 0.5 * dist_norm if you just want to reward "distance < pd"

    
    # Penalize no improvement between consecutive steps
    for i in range(len(prev_distance) - 1):
        pd_curr_norm = prev_distance[i] / GRID_SIZE
        pd_next_norm = prev_distance[i + 1] / GRID_SIZE
        if pd_curr_norm < pd_next_norm:
            #reward -= 0.5 * (pd_next_norm - pd_curr_norm)
            reward -= 1*dist_norm
            # or reward -= 0.5 * dist_norm if you prefer a simpler penalty
    
    """
    for prev_distances in prev_distance:
        if distance < prev_distances:
            reward += 0.5*distance

    
    for i in range(len(prev_distance) - 1):  # Iterate through indices up to the second-to-last element
        if prev_distance[i] < prev_distance[i + 1]:  # Compare current element with the next
            reward -= 0.5 * distance 

    """
    # Collision penalty for identical or very close positions
    for dist in other_robot_distances:
        if dist == (0, 0):  # Check for exact overlap with another robot
            reward -= 0.5  # Apply a negative reward for collision

    # Limit reward to a reasonable range
    #return max(-10, min(1, reward))
    #print(max(-GRID_SIZE, min(reward, GRID_SIZE)))
    return max(-GRID_SIZE, min(reward, GRID_SIZE))

def run_simulation(agent, robots, target, num_steps=1200, epsilon=0.1):
    total_rewards = [0] * len(robots)
    prev_distances = [deque(maxlen=5) for _ in range(len(robots))]

    for _ in range(num_steps):
        target.update_fixed_path()

        for i, robot in enumerate(robots):
            # Step 1: Normalize the state (before action)
            dx, dy = target.x - robot.x, target.y - robot.y
            # Include relative positions of other robots in the state
            other_robot_distances = []
            for j, other_robot in enumerate(robots):
                if j != i:  # Skip itself
                    dist_x = other_robot.x - robot.x
                    dist_y = other_robot.y - robot.y
                    other_robot_distances.append((dist_x, dist_y))
            
            # Flatten other robot distances to include in the state
            flat_other_distances = [coord for dist in other_robot_distances for coord in dist]
            state = normalize_state([dx, dy, robot.vx, robot.vy] + flat_other_distances)

            # Step 2: Choose action and update robot's position
            action = agent.act(state, epsilon)
            ax, ay = ACTION_MAP[action]
            robot.set_velocity(ax, ay)
            robot.update_position()

            # Step 3: Calculate new distance after the action
            dx, dy = target.x - robot.x, target.y - robot.y
            distance = sqrt(dx**2 + dy**2)

            # Update the robot's queue with the new distance
            prev_distances[i].append(distance)

            # Step 4: Compute reward based on the updated distances
            reward = compute_reward(distance, list(prev_distances[i]), i, robots, other_robot_distances)
            total_rewards[i] += reward

            # Step 5: Normalize the next state and update distance
            next_state = normalize_state([dx, dy, robot.vx, robot.vy] + flat_other_distances)

            # Step 6: Remember this transition
            done = distance < 1
            agent.remember(state, action, reward, next_state, done)

    agent.replay()
    return sum(total_rewards)

# Main Training Loop
robots = [Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED) for _ in range(NUM_ROBOTS)]
target = Actor(50, 50, MAX_SPEED)
agent = Agent(STATE_DIM, ACTION_SIZE)

rewards = []
for episode in range(3000):
    epsilon = max(0.1, 1 - episode / 1000)
    # Reinitialize robot positions for each episode if desired:
    # (Optional) For stability, you might reset robots to center:
    # robots = [Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED) for _ in range(NUM_ROBOTS)]
    
    total_reward = run_simulation(agent, robots, target, epsilon=epsilon)
    rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward (All Robots)")
plt.title("Reward Progression with Multiple Robots & Collision Avoidance")
plt.show()

torch.save(agent.policy_network.state_dict(), "MultiRobot/simple_rl_model_multi_robot.pth")
print("Model saved successfully!")