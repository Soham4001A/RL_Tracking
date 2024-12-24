import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt, cos, sin, pi

# Environment Parameters
GRID_SIZE = 1000
TIME_STEP = 0.1
NUM_ROBOTS = 4
NUM_TARGETS = 15
MAX_SPEED = 20
DETECTION_RADIUS = 15 # Fix This
KILL_RADIUS = 2 #Fix This

USE_TRANSFORMER = False  # Set to False to use SimpleNN

# Patrol formation parameters
PATROL_RADIUS = 10
CENTRAL_X, CENTRAL_Y = 50, 50

# State and Action Spaces (Adjust as per your model)
STATE_SIZE = 4 + (NUM_TARGETS * 2) + ((NUM_ROBOTS - 1)*2) + 2  # Include central_obj's position
ACTION_SIZE = 5  # [up, down, left, right, stay]

# Action Maps
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

    def set_velocity(self, vx, vy):
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = np.clip(self.x, 0, GRID_SIZE)
        self.y = np.clip(self.y, 0, GRID_SIZE)

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
        elif i < 4:  # Velocities normalized by MAX_SPEED
            normalized.append(v / MAX_SPEED)
        else:  # Relative positions normalized by GRID_SIZE
            normalized.append(v / GRID_SIZE)
    while len(normalized) < STATE_SIZE:
        normalized.append(0.0)
    return normalized

def get_patrol_positions(central_obj):
    return [
        (central_obj.x + PATROL_RADIUS, central_obj.y),
        (central_obj.x - PATROL_RADIUS, central_obj.y),
        (central_obj.x, central_obj.y + PATROL_RADIUS),
        (central_obj.x, central_obj.y - PATROL_RADIUS)
    ]

def run_simulation(model):
    # Create a central object
    # Define waypoints for the central object (e.g., square path)
    central_waypoints = [(200, 200), (800, 200), (800, 800), (200, 800)]
    central_obj = CentralObject(central_waypoints[0][0], central_waypoints[0][1], max_speed=8, waypoints=central_waypoints)

    # Initialize robots in patrol formation
    robots = [Actor(px, py, MAX_SPEED) for (px, py) in get_patrol_positions(central_obj)]

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
        # Move the central object
        central_obj.update(random_walk=False)  # Set random_walk=True if random movement is desired

        # Update targets' positions
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
            action = choose_action(state, model)
            ax, ay = ACTION_MAP[action]
            robot.set_velocity(ax, ay)
            robot.update_position()

            # Update visualization for this robot
            robot_dots[i].set_data([robot.x], [robot.y])

        # Update visualization for targets
        for i, tar in enumerate(targets):
            target_dots[i].set_data([tar.x], [tar.y])

        # Update central object position (visualization)
        central_dot.set_data([central_obj.x], [central_obj.y])

        return robot_dots + target_dots + [central_dot]

    ani = FuncAnimation(fig, update, init_func=init, frames=200, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    model_path = "/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/Patrol&Protect/multi_robot_patrol_model.pth"
    model = load_model(model_path, STATE_SIZE, ACTION_SIZE)
    run_simulation(model)