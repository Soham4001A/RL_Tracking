import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt, cos, sin, pi, atan2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 10
MIN_DISTANCE = sqrt(2)
VISUALIZATION = True

# RL Parameters
SEQ_LENGTH = 3  # Number of past states to feed into the transformer
STATE_DIM = 6   # [dx, dy, vx, vy, target_vx, target_vy]
ACTION_SIZE = 5  # Expanded actions for diagonal movement
GAMMA = 0.5
LR = 0.0003
CLIP_EPS = 0.25
EPOCHS = 50
BATCH_SIZE = 128*6

# Expanded Action Map (adds diagonal movements)
ACTION_MAP = {
    0: (0, MAX_SPEED),    # Up
    1: (0, -MAX_SPEED),   # Down
    2: (-MAX_SPEED, 0),   # Left
    3: (MAX_SPEED, 0),    # Right
    #4: (MAX_SPEED / sqrt(2), MAX_SPEED / sqrt(2)),    # Up-Right
    #5: (-MAX_SPEED / sqrt(2), MAX_SPEED / sqrt(2)),   # Up-Left
    #6: (MAX_SPEED / sqrt(2), -MAX_SPEED / sqrt(2)),   # Down-Right
    #7: (-MAX_SPEED / sqrt(2), -MAX_SPEED / sqrt(2)),  # Down-Left
    4: (0, 0)             # Stay
}

# Actor class for robot and target
class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed
        self.time = 0

    def update_fixed_path(self, radius=30, center=(50, 50), speed=0.05):
        self.time += speed
        self.vx = -radius * sin(2 * pi * self.time) * speed * 2 * pi
        self.vy = radius * cos(2 * pi * self.time) * speed * 2 * pi
        self.x = center[0] + radius * cos(2 * pi * self.time)
        self.y = center[1] + radius * sin(2 * pi * self.time)

    def set_velocity(self, ax, ay):
        self.vx = np.clip(self.vx + ax * TIME_STEP, -self.max_speed, self.max_speed)
        self.vy = np.clip(self.vy + ay * TIME_STEP, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = max(0, min(self.x, GRID_SIZE))
        self.y = max(0, min(self.y, GRID_SIZE))

# Transformer Network for PPO
class TransformerNetwork(nn.Module):
    def __init__(self, state_size, action_size, d_model=64, nhead=2, dim_feedforward=128, num_layers=2):
        super(TransformerNetwork, self).__init__()
        self.input_layer = nn.Linear(state_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.policy_head = nn.Linear(d_model, action_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_layer(x).transpose(0, 1)
        x = self.transformer(x).transpose(0, 1)[:, -1, :]
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_size):
        self.policy_network = TransformerNetwork(state_dim, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)
        self.memory = []

    def remember(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def act(self, state_seq):
        with torch.no_grad():
            logits, _ = self.policy_network(torch.tensor(state_seq, dtype=torch.float).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        
        # Debugging: Check the values of probs
        if torch.any(probs.isnan()) or torch.any(probs < 0):
            print("Invalid probabilities detected:", probs)
            print("Logits were:", logits)
            raise ValueError("Invalid probabilities in act method.")
        
        return torch.multinomial(probs, 1).item()

    def learn(self):
        if not self.memory:
            return
        # Extract memory into separate components
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions)
        
        # Compute discounted returns
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)  # Insert at the start to reverse the order
        returns = torch.tensor(returns, dtype=torch.float)
        
        for _ in range(EPOCHS):
            logits, values = self.policy_network(states)
            probs = torch.softmax(logits - torch.max(logits), dim=-1)

            # Select log-probabilities of taken actions
            log_probs = torch.log(probs.gather(1, actions.view(-1, 1)).squeeze())

            # Compute advantage
            advantages = returns - values.squeeze()

            # Compute policy loss
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Compute value loss
            value_loss = 0.5 * nn.functional.mse_loss(values.squeeze(), returns)

            # Combine losses
            loss = policy_loss + value_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory after learning
        self.memory = []

def compute_reward(distance, prev_distance, robot_vx, robot_vy, target_vx, target_vy):
    # Distance-based reward
    distance_reduction = prev_distance - distance
    reward = 5 * distance_reduction

    # Velocity matching reward
    velocity_diff = sqrt((robot_vx - target_vx)**2 + (robot_vy - target_vy)**2)
    reward -= 0.5 * velocity_diff  # Penalty for velocity mismatch

    # Small movement penalty
    reward -= 0.05

    # Penalty for being too close
    if distance < MIN_DISTANCE:
        reward -= 5.0

    print(f"Reward is {reward}")
    return reward

def run_simulation(agent, robot, target, num_steps=200):
    state_buffer = deque([[0] * STATE_DIM for _ in range(SEQ_LENGTH)], maxlen=SEQ_LENGTH)
    prev_distance = sqrt((target.x - robot.x)**2 + (target.y - robot.y)**2)
    
    for _ in range(num_steps):
        target.update_fixed_path()
        dx, dy = target.x - robot.x, target.y - robot.y
        distance = sqrt(dx**2 + dy**2)

        state = [dx, dy, robot.vx, robot.vy, target.vx, target.vy]
        state_buffer.append(state)
        action = agent.act(list(state_buffer))
        ax, ay = ACTION_MAP[action]
        robot.set_velocity(ax, ay)
        robot.update_position()

        reward = compute_reward(distance, prev_distance, robot.vx, robot.vy, target.vx, target.vy)
        prev_distance = distance
        agent.remember(list(state_buffer), action, reward, list(state_buffer), False)

    agent.learn()

# Main Training Loop
robot = Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED)
target = Actor(50, 50, MAX_SPEED)
agent = PPOAgent(STATE_DIM, ACTION_SIZE)

for episode in range(100):
    print(f"Episode {episode + 1}")
    run_simulation(agent, robot, target)

torch.save(agent.policy_network.state_dict(), "ppo_transformer_model.pth")
print("Model saved successfully!")