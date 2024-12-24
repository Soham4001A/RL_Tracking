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
USE_TRANSFORMER = False  # Set to False to use SimpleNN
EPISODES = 10000

# Patrol Formation Parameters
PATROL_RADIUS = sqrt((2)**2+(2)**2)

# Target Parameters
NUM_ROBOTS = 4
NUM_TARGETS = 2
DETECTION_RADIUS = 10 #This needs to be based off the central obj
KILL_RADIUS = sqrt((5)**2+(5)**2) # This needs to be based off the central obj

# RL Parameters
# State: (robot_x, robot_y, vx, vy) + 2 targets (dx, dy each) + 3 other robots (dx, dy each)
STATE_DIM = 4 + (NUM_TARGETS * 2) + ((NUM_ROBOTS - 1) * 2) + 2 
ACTION_SIZE = 5
GAMMA = 0.9
LR = 0.0005
BATCH_SIZE = 128

# Action Maps
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
    def __init__(self, waypoints, max_speed):
        """
        Initialize the target with a set of waypoints and a maximum speed.
        
        Args:
            waypoints (list of tuples): Sequence of waypoints (x, y) the target will follow.
            max_speed (float): Maximum speed of the target.
        """
        self.waypoints = waypoints  # List of waypoints
        self.current_waypoint_idx = 0  # Start at the first waypoint
        self.x, self.y = self.waypoints[self.current_waypoint_idx]  # Start at the first waypoint
        self.max_speed = max_speed

    def move_to_next_waypoint(self):
        """
        Move toward the current waypoint. If the waypoint is reached, advance to the next waypoint.
        """
        # Current waypoint coordinates
        target_x, target_y = self.waypoints[self.current_waypoint_idx]

        # Compute direction vector to the waypoint
        direction_x = target_x - self.x
        direction_y = target_y - self.y
        distance_to_waypoint = sqrt(direction_x**2 + direction_y**2)

        # Check if we reached the waypoint
        if distance_to_waypoint < 0.1:  # Small threshold to consider the waypoint "reached"
            # Advance to the next waypoint
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            target_x, target_y = self.waypoints[self.current_waypoint_idx]  # Update to new waypoint
            direction_x = target_x - self.x
            direction_y = target_y - self.y
            distance_to_waypoint = sqrt(direction_x**2 + direction_y**2)

        # Normalize direction and move toward the waypoint
        if distance_to_waypoint > 0:
            norm_x = direction_x / distance_to_waypoint
            norm_y = direction_y / distance_to_waypoint
            self.x += norm_x * self.max_speed * TIME_STEP
            self.y += norm_y * self.max_speed * TIME_STEP

    def update(self):
        """Update the target's position."""
        self.move_to_next_waypoint()

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

class TransformerNN(nn.Module):
    def __init__(self, state_dim, action_size, num_heads=4, num_layers=2):
        super(TransformerNN, self).__init__()
        self.state_dim = state_dim

        # Input Embedding
        self.input_embedding = nn.Linear(state_dim, 64)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, state_dim, 64))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=num_heads, dim_feedforward=128
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        # Add positional encoding to input embedding
        x = self.input_embedding(x) + self.positional_encoding[:, :x.size(1), :]
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate features (mean pooling) and pass through dense layers
        x = torch.mean(x, dim=1)
        return self.fc(x)
    
class Agent:
    def __init__(self, state_dim, action_size):
        if USE_TRANSFORMER:
            self.policy_network = TransformerNN(state_dim, action_size)
        else:
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
        if i < 2:  # Positions
            normalized.append(v / GRID_SIZE)
        elif i < 4:  # Velocities
            normalized.append(v / MAX_SPEED)
        else:  # Relative positions
            normalized.append(v / GRID_SIZE)
    
    # Pad the state with zeros to match STATE_DIM if necessary
    while len(normalized) < STATE_DIM:
        normalized.append(0.0)
    
    return normalized

def compute_reward(curr_patrol_dist, prev_patrol_dist, robots, other_robot_distances):
    """
    Compute the reward for a robot based only on its patrol behavior and avoiding collisions.
    """
    # Base reward: closer to patrol position is better
    #eward = 0
    reward = -curr_patrol_dist / GRID_SIZE
    if prev_patrol_dist is not None and curr_patrol_dist < prev_patrol_dist:
        reward += 0.5  # Reward for improving patrol position

    # Collision penalty: Penalize for being too close to other robots
    for dist in other_robot_distances:
        if dist == (0, 0):  # Collision or very close proximity
            reward -= 0.5  # Strong penalty for collisions
    """
    if detection (robot enteres detection radius):
        set reward to 0
        add reward for reducing distance to targets for 1 robot currently closest to target
        penalize if more than 1 robot intercepts the target (keep the rest on patrol)
    """
    # Limit reward to a reasonable range
    return max(-10, min(1, reward))

def get_patrol_positions(central_obj):
    return [
        (central_obj.x + PATROL_RADIUS, central_obj.y),
        (central_obj.x - PATROL_RADIUS, central_obj.y),
        (central_obj.x, central_obj.y + PATROL_RADIUS),
        (central_obj.x, central_obj.y - PATROL_RADIUS)
    ]

def run_simulation(agent, robots, targets, central_obj, num_steps=200, epsilon=0.1):
    """
    Run the patrol simulation where robots only patrol the central object.
    Targets remain in the simulation for dynamics but do not affect rewards.
    """
    total_rewards = [0] * len(robots)
    prev_patrol_distances = [None] * len(robots)

    for step in range(num_steps):
        # Update target positions (purely for dynamics)
        for t in targets:
            t.update()

        # Dynamically compute patrol positions based on central_obj's current position
        PATROL_POSITIONS = get_patrol_positions(central_obj)

        for i, robot in enumerate(robots):

            # Compute distances to other robots
            other_robot_distances = [
                sqrt((other_robot.x - robot.x)**2 + (other_robot.y - robot.y)**2)
                for j, other_robot in enumerate(robots) if j != i
            ]

            # Create state
            state = normalize_state([
                robot.x, robot.y, robot.vx, robot.vy,
                central_obj.x, central_obj.y
            ] + [
                other_robot.x - robot.x for other_robot in robots if other_robot != robot
            ] + [
                other_robot.y - robot.y for other_robot in robots if other_robot != robot
            ] + [
                target.x - robot.x for target in targets
            ] + [
                target.y - robot.y for target in targets
            ])

            # Choose action and update robot's position
            action = agent.act(state, epsilon)
            ax, ay = ACTION_MAP[action]
            robot.set_velocity(ax, ay)
            robot.update_position()
            
            # Compute patrol distance
            desired_x, desired_y = PATROL_POSITIONS[i]
            curr_patrol_dist = sqrt((robot.x - desired_x)**2 + (robot.y - desired_y)**2)

            # Compute reward based on patrol behavior only
            reward = compute_reward(curr_patrol_dist, prev_patrol_distances[i], robots, other_robot_distances)
            total_rewards[i] += reward

            # Update patrol distance for the next step
            prev_patrol_distances[i] = curr_patrol_dist

        # Replay experiences for learning
        agent.replay()

    return sum(total_rewards)

# Main Training Loop
central_obj = CentralObject(50, 50)
agent = Agent(STATE_DIM, ACTION_SIZE)

rewards = []
for episode in range(EPISODES):
    epsilon = max(0.1, 1 - episode / 1000)

    # Dynamically compute patrol positions
    PATROL_POSITIONS = get_patrol_positions(central_obj)

    # Reset robots and targets for each episode based on current patrol positions
    robots = [Actor(x, y, MAX_SPEED) for (x, y) in PATROL_POSITIONS]
    waypoints_target_1 = [(48, 48), (48, 82), (12, 82), (12, 48)]
    waypoints_target_2 = [(31, 51), (91, 51), (91, 13), (31, 13)]
    targets = [
        AdversarialTarget(waypoints=waypoints_target_1, max_speed=MAX_SPEED),
        AdversarialTarget(waypoints=waypoints_target_2, max_speed=MAX_SPEED)
    ]

    # Run simulation
    total_reward = run_simulation(agent, robots, targets, central_obj, epsilon=epsilon)
    rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Plot results
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward (All Robots)")
plt.title("Reward Progression for Patrol Task")
plt.show()

torch.save(agent.policy_network.state_dict(), "multi_robot_patrol_model.pth")
print("Model saved successfully!")