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

# Patrol Formation Parameters
PATROL_RADIUS = 4
PATROL_POSITIONS = [
    (50 + PATROL_RADIUS, 50),
    (50 - PATROL_RADIUS, 50),
    (50, 50 + PATROL_RADIUS),
    (50, 50 - PATROL_RADIUS)
]

# Target Parameters
NUM_ROBOTS = 4
NUM_TARGETS = 2
DETECTION_RADIUS = 15
KILL_RADIUS = 2

# RL Parameters
# State: (robot_x, robot_y, vx, vy) + 2 targets (dx, dy each) + 3 other robots (dx, dy each)
STATE_DIM = 4 + (NUM_TARGETS * 2) + ((NUM_ROBOTS - 1) * 2)
ACTION_SIZE = 5
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

class CentralObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class AdversarialTarget:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.max_speed = max_speed

    def update_random(self):
        angle = np.random.uniform(0, 2*np.pi)
        speed = np.random.uniform(0, self.max_speed)
        self.x += speed * cos(angle) * TIME_STEP
        self.y += speed * sin(angle) * TIME_STEP
        self.x = np.clip(self.x, 0, GRID_SIZE)
        self.y = np.clip(self.y, 0, GRID_SIZE)

class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed

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

def normalize_state(vals):
    # Normalize positions by GRID_SIZE and velocities by MAX_SPEED
    normalized = []
    for i, v in enumerate(vals):
        if i % 2 == 0 or i == 0: # if dealing with an x or dx
            normalized.append(v / GRID_SIZE)
        else:
            normalized.append(v / GRID_SIZE)
    return normalized

def compute_reward(robot, robot_idx, robots, targets, central_obj, 
                   prev_patrol_dist, prev_target_dist):
    """
    Compute the reward for a single robot at the current timestep.
    
    Args:
        robot: The current robot being evaluated.
        robot_idx: Index of the robot.
        robots: List of all robots (including the current one).
        targets: List of adversarial targets.
        central_obj: The central object.
        prev_patrol_dist: The robot's patrol distance at the previous timestep.
        prev_target_dist: The robot's closest target distance at the previous timestep.
    
    Returns:
        reward: A floating point reward value.
        curr_patrol_dist: The current patrol distance (to be stored for the next step).
        curr_target_dist: The current closest target distance (to be stored for the next step).
    """
    # Patrol behavior: Encourage staying near the patrol position
    desired_x, desired_y = PATROL_POSITIONS[robot_idx]
    curr_patrol_dist = sqrt((robot.x - desired_x)**2 + (robot.y - desired_y)**2)

    # Reward for staying near the patrol position
    if curr_patrol_dist < PATROL_RADIUS:
        reward = 1.0  # Strong reward for being close to patrol position
    else:
        # Penalty proportional to distance away from patrol position
        reward = - (curr_patrol_dist / GRID_SIZE)

    # If the robot moved closer to its patrol position, add a small incremental reward
    if prev_patrol_dist is not None and curr_patrol_dist < prev_patrol_dist:
        reward += 0.2  # Encourage incremental improvement

    # Penalize unnecessary movement if no target is nearby
    if curr_patrol_dist > PATROL_RADIUS:
        reward -= (robot.vx**2 + robot.vy**2) * 0.01  # Small penalty for high velocity

    # Target behavior: Encourage chasing only if the target is near the central object
    min_target_dist = float('inf')
    target_near_central = False
    cx, cy = central_obj.x, central_obj.y

    for t in targets:
        dist_to_robot = sqrt((t.x - robot.x)**2 + (t.y - robot.y)**2)
        dist_to_central = sqrt((t.x - cx)**2 + (t.y - cy)**2)

        if dist_to_robot < min_target_dist:
            min_target_dist = dist_to_robot

        # Check if target is within the detection radius of the central object
        if dist_to_central < DETECTION_RADIUS:
            target_near_central = True

    curr_target_dist = min_target_dist

    if target_near_central:
        # Reward movement toward the target if it's near the central object
        reward += - (curr_target_dist / GRID_SIZE)

        # Additional reward for moving closer to the target
        if prev_target_dist is not None and curr_target_dist < prev_target_dist:
            reward += 0.5

        # If robot intercepts the target (within KILL_RADIUS), give a large reward
        if curr_target_dist < KILL_RADIUS:
            reward += 5.0  # Strong reward for successful interception
    else:
        # Penalize moving away from the patrol position when no target is near
        reward -= (curr_patrol_dist / GRID_SIZE) * 0.5

    # Collision penalty: Check distance to other robots
    for j, other_robot in enumerate(robots):
        if j != robot_idx:
            dist = sqrt((other_robot.x - robot.x)**2 + (other_robot.y - robot.y)**2)
            if dist < 1:  # Collision threshold
                reward -= 2.0  # Strong penalty for collisions

    # Optional: Clip reward if desired (to prevent extreme values)
    reward = max(min(reward, 5.0), -5.0)

    return reward, curr_patrol_dist, curr_target_dist

def run_simulation(agent, robots, targets, central_obj, num_steps=200, epsilon=0.1):
    total_rewards = [0] * len(robots)
    
    # Initialize previous distances for each robot
    prev_patrol_distances = [None] * len(robots)
    prev_target_distances = [None] * len(robots)

    for step in range(num_steps):
        # Update targets positions
        for t in targets:
            t.update_random()

        for i, robot in enumerate(robots):
            # Create state
            # Robot states: (x, y, vx, vy)
            # Targets: (dx_t, dy_t) for each target
            # Other robots: (dx_r, dy_r) for each other robot
            dx_targets = []
            for tar in targets:
                dx_targets.append(tar.x - robot.x)
                dx_targets.append(tar.y - robot.y)

            dx_robots = []
            for j, other_robot in enumerate(robots):
                if j != i:
                    dx_robots.append(other_robot.x - robot.x)
                    dx_robots.append(other_robot.y - robot.y)

            state = [robot.x, robot.y, robot.vx, robot.vy] + dx_targets + dx_robots
            state = normalize_state(state)

            # Act
            action = agent.act(state, epsilon)
            ax, ay = ACTION_MAP[action]
            robot.set_velocity(ax, ay)
            robot.update_position()

        # After all robots move, compute rewards and update distances
        for i, robot in enumerate(robots):
            # Compute reward
            reward, curr_patrol_dist, curr_target_dist = compute_reward(
                robot, 
                i, 
                robots, 
                targets, 
                central_obj, 
                prev_patrol_distances[i], 
                prev_target_distances[i]
            )
            
            total_rewards[i] += reward

            # Update previous distances for the next timestep
            prev_patrol_distances[i] = curr_patrol_dist
            prev_target_distances[i] = curr_target_dist

        # At the end of the step, we replay once
        agent.replay()

    return sum(total_rewards)

# Main Training Loop
central_obj = CentralObject(50, 50)
agent = Agent(STATE_DIM, ACTION_SIZE)

rewards = []
for episode in range(10000):
    # Reinitialize the environment if desired
    robots = [Actor(x, y, MAX_SPEED) for (x, y) in PATROL_POSITIONS]
    targets = [AdversarialTarget(np.random.uniform(0, GRID_SIZE), np.random.uniform(0, GRID_SIZE), MAX_SPEED) for _ in range(NUM_TARGETS)]

    epsilon = max(0.1, 1 - episode / 1000)
    total_reward = run_simulation(agent, robots, targets, central_obj, epsilon=epsilon)
    rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward (All Robots)")
plt.title("Reward Progression with Patrol and Intercept Task")
plt.show()

torch.save(agent.policy_network.state_dict(), "multi_robot_patrol_model.pth")
print("Model saved successfully!")