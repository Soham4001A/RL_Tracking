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
ACTION_SIZE = 5  # Actions: Up, Down, Left, Right
GAMMA = 0.9
LR = 0.0005
BATCH_SIZE = 128

# Action Map
ACTION_MAP = {
    0: (0, MAX_SPEED),    # Up
    1: (0, -MAX_SPEED),   # Down
    2: (-MAX_SPEED, 0),   # Left
    3: (MAX_SPEED, 0),    # Right
    4: (0,0)              # Stay 
}

class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed
        self.time = 0

    def update_fixed_path(self, side_length=30, center=(50, 50), speed=0.05):
        # Square path for the target
        perimeter = 4 * side_length
        self.time += speed * TIME_STEP
        distance_along_perimeter = (self.time * perimeter) % perimeter

        if distance_along_perimeter < side_length:
            # Top side: move right
            self.x = center[0] - side_length / 2 + distance_along_perimeter
            self.y = center[1] - side_length / 2
        elif distance_along_perimeter < 2 * side_length:
            # Right side: move down
            self.x = center[0] + side_length / 2
            self.y = center[1] - side_length / 2 + (distance_along_perimeter - side_length)
        elif distance_along_perimeter < 3 * side_length:
            # Bottom side: move left
            self.x = center[0] + side_length / 2 - (distance_along_perimeter - 2 * side_length)
            self.y = center[1] + side_length / 2
        else:
            # Left side: move up
            self.x = center[0] - side_length / 2
            self.y = center[1] + side_length / 2 - (distance_along_perimeter - 3 * side_length)

    def set_velocity(self, ax, ay):
        self.vx = np.clip(ax, -self.max_speed, self.max_speed)
        self.vy = np.clip(ay, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = max(0, min(self.x, GRID_SIZE))
        self.y = max(0, min(self.y, GRID_SIZE))

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

class Agent:
    def __init__(self, state_dim, action_size):
        self.policy_network = SimpleNN(state_dim, action_size)
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

def compute_reward(distance, prev_distance, robot_idx, robots):
    # Base reward: closer distance is better
    reward = -distance / GRID_SIZE
    if distance < prev_distance:
        reward += 0.5

    # Collision penalty
    # If robot i is too close to any other robot, penalize
    """
    for j, other_robot in enumerate(robots):
        if j != robot_idx:
            dist_to_other = sqrt((robots[robot_idx].x - other_robot.x)**2 + (robots[robot_idx].y - other_robot.y)**2)
            if dist_to_other < 2:  # Collision threshold
                reward -= 1.0
                break
    """
    return max(-1, min(1, reward))

def run_simulation(agent, robots, target, num_steps=200, epsilon=0.1):
    total_rewards = [0] * len(robots)
    prev_distances = [float('inf')] * len(robots)

    for _ in range(num_steps):
        target.update_fixed_path()

        # For each robot, choose action and update
        for i, robot in enumerate(robots):
            dx, dy = target.x - robot.x, target.y - robot.y
            distance = sqrt(dx**2 + dy**2)

            state = normalize_state([dx, dy, robot.vx, robot.vy])
            action = agent.act(state, epsilon)
            ax, ay = ACTION_MAP[action]
            robot.set_velocity(ax, ay)
            robot.update_position()

            next_state = normalize_state([target.x - robot.x, target.y - robot.y, robot.vx, robot.vy])
            reward = compute_reward(distance, prev_distances[i], i, robots)
            prev_distances[i] = distance
            total_rewards[i] += reward

            done = distance < 1
            agent.remember(state, action, reward, next_state, done)

    agent.replay()
    # Return the sum of all rewards as a single metric
    return sum(total_rewards)

# Main Training Loop
NUM_ROBOTS = 4
robots = [Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED) for _ in range(NUM_ROBOTS)]
target = Actor(50, 50, MAX_SPEED)
agent = Agent(STATE_DIM, ACTION_SIZE)

rewards = []
for episode in range(10000):
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

torch.save(agent.policy_network.state_dict(), "simple_rl_model_multi_robot.pth")
print("Model saved successfully!")