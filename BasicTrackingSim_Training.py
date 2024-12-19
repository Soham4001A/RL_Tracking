import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt, cos, sin, pi
from collections import deque
import matplotlib.pyplot as plt

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 10
VISUALIZATION = False

# RL Parameters
STATE_DIM = 4  # [dx, dy, robot_vx, robot_vy]
ACTION_SIZE = 4  # Actions: Up, Down, Left, Right, Stay
GAMMA = 0.9
LR = 0.0005  # Reduced learning rate for stability
BATCH_SIZE = 128  # Larger batch size for gradient stability

# Action Map
ACTION_MAP = {
    0: (0, MAX_SPEED),    # Up
    1: (0, -MAX_SPEED),   # Down
    2: (-MAX_SPEED, 0),   # Left
    3: (MAX_SPEED, 0),    # Right
    #4: (0, 0)             # Stay
}

# Actor class for robot and target
class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed

    def update_fixed_path(self, radius=30, center=(50, 50), speed=0.01):  # Slower target
        angle = 2 * pi * speed
        self.x = center[0] + radius * cos(angle)
        self.y = center[1] + radius * sin(angle)

    def set_velocity(self, ax, ay):
        self.vx = np.clip(ax, -self.max_speed, self.max_speed)
        self.vy = np.clip(ay, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = max(0, min(self.x, GRID_SIZE))
        self.y = max(0, min(self.y, GRID_SIZE))

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, state_dim, action_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# RL Agent
class Agent:
    def __init__(self, state_dim, action_size):
        self.policy_network = SimpleNN(state_dim, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)
        self.memory = deque(maxlen=5000)  # Increased buffer size

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

        # Compute predictions
        q_values = self.policy_network(states)
        predictions = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = nn.functional.mse_loss(predictions, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1)  # Gradient clipping
        self.optimizer.step()

# Reward Function
def compute_reward(distance, prev_distance):
    reward = -distance / GRID_SIZE  # Normalize reward
    if distance < prev_distance:
        reward += 0.5  # Small positive reward for reducing distance
    return max(-1, min(1, reward))  # Clip reward

# Normalize state
def normalize_state(state):
    return [s / GRID_SIZE for s in state]

# Simulation Function
def run_simulation(agent, robot, target, num_steps=200, epsilon=0.1):
    total_reward = 0
    prev_distance = float('inf')
    for _ in range(num_steps):
        target.update_fixed_path()
        dx, dy = target.x - robot.x, target.y - robot.y
        distance = sqrt(dx**2 + dy**2)

        state = normalize_state([dx, dy, robot.vx, robot.vy])
        action = agent.act(state, epsilon)
        ax, ay = ACTION_MAP[action]
        robot.set_velocity(ax, ay)
        robot.update_position()

        next_state = normalize_state([target.x - robot.x, target.y - robot.y, robot.vx, robot.vy])
        reward = compute_reward(distance, prev_distance)
        prev_distance = distance
        total_reward += reward

        done = distance < 1  # Episode ends if robot reaches target
        agent.remember(state, action, reward, next_state, done)

        if done:
            break

    agent.replay()
    return total_reward

# Main Training Loop
robot = Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED)
target = Actor(50, 50, MAX_SPEED)
agent = Agent(STATE_DIM, ACTION_SIZE)

rewards = []
for episode in range(15000):
    epsilon = max(0.1, 1 - episode / 1000)  # Slower epsilon decay
    total_reward = run_simulation(agent, robot, target, epsilon=epsilon)
    rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Plot rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Progression")
plt.show()

torch.save(agent.policy_network.state_dict(), "simple_rl_model.pth")
print("Model saved successfully!")