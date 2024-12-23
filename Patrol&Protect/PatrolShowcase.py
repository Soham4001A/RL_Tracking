import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt, cos, sin, pi

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
NUM_ROBOTS = 4
NUM_TARGETS = 2
MAX_SPEED = 20
DETECTION_RADIUS = 15
KILL_RADIUS = 2

USE_TRANSFORMER = False  # Set to False to use SimpleNN

# Patrol formation parameters
PATROL_RADIUS = 10
CENTRAL_X, CENTRAL_Y = 50, 50
PATROL_POSITIONS = [
    (CENTRAL_X + PATROL_RADIUS, CENTRAL_Y),
    (CENTRAL_X - PATROL_RADIUS, CENTRAL_Y),
    (CENTRAL_X, CENTRAL_Y + PATROL_RADIUS),
    (CENTRAL_X, CENTRAL_Y - PATROL_RADIUS)
]

# State and Action Spaces (Adjust as per your model)
# Example state: [robot_x, robot_y, vx, vy] + two targets positions (dx, dy) each + other robots
# Make sure you align this with how your model was trained
# Here we assume a placeholder state size and action space similar to previous code
STATE_SIZE = 4 + (NUM_TARGETS * 2) + ((NUM_ROBOTS - 1)*2)  # Adjust if needed
ACTION_SIZE = 5  # [up, down, left, right, stay]
ACTIONS = {
    0: (0, MAX_SPEED),    # up
    1: (0, -MAX_SPEED),   # down
    2: (-MAX_SPEED, 0),   # left
    3: (MAX_SPEED, 0),    # right
    4: (0, 0)             # stay
}

class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed

    def set_velocity(self, vx, vy):
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = np.clip(self.x, 0, GRID_SIZE)
        self.y = np.clip(self.y, 0, GRID_SIZE)

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

class UpdatedModel(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(UpdatedModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.fc(state)

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

def load_model(model_path, state_size, action_size):
    if USE_TRANSFORMER:
        model = TransformerNN(state_size, action_size)
    else:
        model = UpdatedModel(state_size, action_size)  # UpdatedModel corresponds to SimpleNN
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"{'TransformerNN' if USE_TRANSFORMER else 'SimpleNN'} model weights loaded successfully!")
    return model

def choose_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

def normalize_state(vals):
    normalized = []
    for i, v in enumerate(vals):
        if i < 2:  # Positions normalized by GRID_SIZE
            normalized.append(v / GRID_SIZE)
        else:  # Velocities normalized by MAX_SPEED
            normalized.append(v / MAX_SPEED)
    while len(normalized) < STATE_SIZE:
        normalized.append(0.0)
    return normalized

def run_simulation(model):
    # Create a central object
    central_obj = CentralObject(CENTRAL_X, CENTRAL_Y)

    # Initialize robots in patrol formation
    robots = [Actor(px, py, MAX_SPEED) for (px, py) in PATROL_POSITIONS]

    # Define waypoints for Target 1 (around 48, 48) and Target 2 (around 51, 51)
    waypoints_target_1 = [(48, 48), (48, 82), (12, 82), (12, 48)]
    waypoints_target_2 = [(31, 51), (91, 51), (91, 13), (31, 13)]

    # Initialize the adversarial targets
    targets = [
        AdversarialTarget(waypoints=waypoints_target_1, max_speed=MAX_SPEED),
        AdversarialTarget(waypoints=waypoints_target_2, max_speed=MAX_SPEED)
    ]

    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_title("Patrol & Protect Showcase")

    # Plot elements
    # Central Object: Red Star
    central_dot, = ax.plot([central_obj.x], [central_obj.y], 'r*', markersize=10, label='Central Object')

    # Robots
    colors = ['bo', 'go', 'ro', 'mo']
    robot_dots = []
    for c in colors:
        dot, = ax.plot([], [], c)
        robot_dots.append(dot)

    # Targets: Yellow circles
    target_dots = []
    for _ in range(NUM_TARGETS):
        tdot, = ax.plot([], [], 'yo', label='Adversarial Target')
        target_dots.append(tdot)

    ax.legend()

    def init():
        for rd in robot_dots:
            rd.set_data([], [])
        for td in target_dots:
            td.set_data([], [])
        return robot_dots + target_dots + [central_dot]

    def update(frame):
        # Update targets' positions (following their square patterns)
        for t in targets:
            t.update()

        for i, robot in enumerate(robots):
            # Compute state
            # State: robot_x, robot_y, vx, vy, targets(dx, dy for each), other_robots(dx, dy for each)
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
            state = normalize_state([robot.x, robot.y, robot.vx, robot.vy] + [
                other_robot.x - robot.x for other_robot in robots if other_robot != robot
            ] + [
                other_robot.y - robot.y for other_robot in robots if other_robot != robot
            ])

            # Choose action from model
            action = choose_action(state, model)
            vx, vy = ACTIONS[action]
            robot.set_velocity(vx, vy)
            robot.update_position()

        # Update visualization
        for i, robot in enumerate(robots):
            robot_dots[i].set_data([robot.x], [robot.y])

        for i, tar in enumerate(targets):
            target_dots[i].set_data([tar.x], [tar.y])

        return robot_dots + target_dots + [central_dot]

    ani = FuncAnimation(fig, update, init_func=init, frames=200, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    model_path = "/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/Patrol&Protect/multi_robot_patrol_model.pth"
    model = load_model(model_path, STATE_SIZE, ACTION_SIZE)
    run_simulation(model)