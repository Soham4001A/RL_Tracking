import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt, cos, sin, pi
from collections import deque
import matplotlib.pyplot as plt

# Environment Parameters
GRID_SIZE = 1000
TIME_STEP = 0.1
MAX_SPEED = 10
USE_TRANSFORMER = False  # Set to False to use SimpleNN
EPISODES = 10000

# Patrol Formation Parameters
PATROL_RADIUS = sqrt((2)**2+(2)**2)

# Central Obj Movement Control
CENTRAL_OBJ_RANDOM_WALK = False
CENTRAL_OBJ_WAYPOINT = True

# Target Parameters
NUM_ROBOTS = 4
NUM_TARGETS = 15
DETECTION_RADIUS = PATROL_RADIUS * 1.5  # Scaled to central object size
KILL_RADIUS = PATROL_RADIUS * 0.75  # Scaled to patrol radius

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
    def __init__(self, x, y, max_speed=5, waypoints=None):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed
        self.waypoints = waypoints or []
        self.current_waypoint_idx = 0

    def move_to_next_waypoint(self):
        """Move toward the current waypoint."""
        if not self.waypoints:
            return

        target_x, target_y = self.waypoints[self.current_waypoint_idx]
        direction_x = target_x - self.x
        direction_y = target_y - self.y
        distance = sqrt(direction_x**2 + direction_y**2)

        # If close to the waypoint, move to the next one
        if distance < 1:
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
        else:
            norm_x = direction_x / distance
            norm_y = direction_y / distance
            self.set_velocity(norm_x * self.max_speed, norm_y * self.max_speed)

        # Update position based on the set velocity
        self.update_position()

    def random_walk(self):
        """Perform a random walk."""
        if np.random.rand() < 0.1:  # 10% chance to change direction
            vx = np.random.uniform(-self.max_speed, self.max_speed)
            vy = np.random.uniform(-self.max_speed, self.max_speed)
            self.set_velocity(vx, vy)
        self.update_position()

    def set_velocity(self, vx, vy):
        """Set the velocity of the central object."""
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

    def update_position(self):
        """Update the position of the central object."""
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = np.clip(self.x, 0, GRID_SIZE)
        self.y = np.clip(self.y, 0, GRID_SIZE)

    def update(self, random_walk=False):
        """Update the central object's position."""
        if random_walk:
            self.random_walk()
        else:
            self.move_to_next_waypoint()

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

def compute_reward_old(curr_patrol_dist, prev_patrol_dist, robots, other_robot_distances):
    """
    Compute the reward for a singular robot 
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

def compute_reward(curr_patrol_dist, prev_patrol_dist, robots, other_robot_distances, detection_radius, target_distances):
    """
    Compute the reward for a robot based on patrol behavior, collision avoidance, 
    and interaction with targets.
    
    Args:
        curr_patrol_dist (float): Current distance from the patrol position.
        prev_patrol_dist (float): Previous distance from the patrol position.
        robots (list): List of all robots in the environment.
        other_robot_distances (list): Distances to other robots.
        detection_radius (float): Detection radius for target interaction.
        target_distances (list): Distances to all targets.
    
    Returns:
        float: Computed reward for the robot.
    """
    reward = 0

    # Reward for maintaining or improving patrol position
    reward -= curr_patrol_dist / GRID_SIZE  # Base penalty for being far from patrol position
    if prev_patrol_dist is not None and curr_patrol_dist < prev_patrol_dist:
        reward += 1.0  # Bonus for moving closer to patrol position

    # Collision penalty: Penalize for being too close to other robots
    for dist in other_robot_distances:
        if dist < KILL_RADIUS:
            reward -= 2.0  # Strong penalty for collisions

    # Interaction with targets: Reward for entering detection radius
    closest_target_dist = min(target_distances)
    if closest_target_dist < detection_radius:
        reward += 1.5  # Bonus for entering detection radius
        # Additional bonus for being the closest robot to a target
        if closest_target_dist == min(target_distances):  # Simplified line
            reward += 0.5  # Additional reward for closest proximity to target

    # Penalty for over-convergence on targets
    num_robots_in_radius = sum(1 for dist in target_distances if dist < detection_radius)
    if num_robots_in_radius > 1:
        reward -= 0.5 * (num_robots_in_radius - 1)  # Penalize overcrowding

    # Penalty for idleness (not moving significantly or staying far from patrol)
    movement_penalty = 1.0 - (curr_patrol_dist / GRID_SIZE)
    reward -= movement_penalty * 0.1  # Small penalty for idleness

    # Limit reward to a reasonable range
    reward = max(-10, min(5, reward))
    return reward

def get_patrol_positions(central_obj):
    return [
        (central_obj.x + PATROL_RADIUS, central_obj.y),
        (central_obj.x - PATROL_RADIUS, central_obj.y),
        (central_obj.x, central_obj.y + PATROL_RADIUS),
        (central_obj.x, central_obj.y - PATROL_RADIUS)
    ]

def run_simulation(agent, robots, targets, central_obj, num_steps=200, epsilon=0.1):
    total_rewards = [0] * len(robots)
    prev_patrol_distances = [None] * len(robots)

    for step in range(num_steps):
        # Move the central object
        central_obj.update()

        # Update target positions
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
            reward = compute_reward(
                curr_patrol_dist=curr_patrol_dist,
                prev_patrol_dist=prev_patrol_distances[i],
                robots=robots,
                other_robot_distances=other_robot_distances,
                detection_radius=DETECTION_RADIUS,
                target_distances=[sqrt((target.x - robot.x)**2 + (target.y - robot.y)**2) for target in targets]
            )
            total_rewards[i] += reward

            # Update patrol distance for the next step
            prev_patrol_distances[i] = curr_patrol_dist

        # Replay experiences for learning
        agent.replay()

    return sum(total_rewards)

# Main Training Loop
central_waypoints = [(200, 200), (800, 200), (800, 800), (200, 800)]  # Square path
central_obj = CentralObject(central_waypoints[0][0], central_waypoints[0][1], max_speed=5, waypoints=central_waypoints)
agent = Agent(STATE_DIM, ACTION_SIZE)

rewards = []
for episode in range(EPISODES):
    epsilon = max(0.1, 1 - episode / 1000)

    # Dynamically compute patrol positions
    PATROL_POSITIONS = get_patrol_positions(central_obj)

    # Reset robots and targets for each episode based on current patrol positions
    robots = [Actor(x, y, MAX_SPEED) for (x, y) in PATROL_POSITIONS]
    # Define updated waypoints for 15 targets
    waypoints_targets = [
        # Target 1: Circular path (centered at 500, 500 with a radius of 100)
        [(500 + 100 * cos(i * 2 * pi / 8), 500 + 100 * sin(i * 2 * pi / 8)) for i in range(8)],

        # Target 2: Square path (bottom-left corner at 200, 200 with a side length of 200)
        [(200, 200), (200, 400), (400, 400), (400, 200)],

        # Target 3: Zig-zag path (vertical zig-zag in the left quadrant)
        [(150, 100), (200, 300), (150, 500), (200, 700), (150, 900)],

        # Target 4: Random walk path (scattered points in the grid)
        [(np.random.randint(100, 900), np.random.randint(100, 900)) for _ in range(5)],

        # Target 5: Diagonal line (from top-left to bottom-right quadrant)
        [(i, i) for i in range(100, 900, 200)],

        # Target 6: Figure 8 path (centered at 500, 500 with alternating radii)
        [(500 + 100 * cos(i * pi / 4), 500 + 50 * sin(i * pi / 4) * (1 if i % 2 == 0 else -1)) for i in range(8)],

        # Target 7: Small circle (centered at 300, 300 with a radius of 50)
        [(300 + 50 * cos(i * 2 * pi / 8), 300 + 50 * sin(i * 2 * pi / 8)) for i in range(8)],

        # Target 8: Larger square (bottom-left corner at 600, 600 with a side length of 300)
        [(600, 600), (600, 900), (900, 900), (900, 600)],

        # Target 9: Vertical zig-zag (centered at 700, vertical axis)
        [(700, 100), (700, 300), (700, 500), (700, 700), (700, 900)],

        # Target 10: Converging inward (diagonal inward from corners)
        [(i, 1000 - i) for i in range(100, 900, 200)],

        # Target 11: Horizontal zig-zag (centered at 500, horizontal axis)
        [(100, 500), (300, 500), (500, 500), (700, 500), (900, 500)],

        # Target 12: Elliptical path (centered at 500, 500 with radii 150 and 100)
        [(500 + 150 * cos(i * 2 * pi / 8), 500 + 100 * sin(i * 2 * pi / 8)) for i in range(8)],

        # Target 13: Figure 8 path (smaller scale centered at 400, 400)
        [(400 + 50 * cos(i * pi / 4), 400 + 30 * sin(i * pi / 4) * (1 if i % 2 == 0 else -1)) for i in range(8)],

        # Target 14: Static path (stationary at 800, 800)
        [(800, 800)],

        # Target 15: Random walk path (small random walk in the center)
        [(np.random.randint(450, 550), np.random.randint(450, 550)) for _ in range(5)],
    ]
    targets = [
    AdversarialTarget(waypoints=waypoints_targets[i], max_speed=MAX_SPEED)
    for i in range(NUM_TARGETS)
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