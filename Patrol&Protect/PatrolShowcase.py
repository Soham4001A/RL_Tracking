import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt, cos, sin, pi
from shared_utils import CentralObject, AdversarialTarget, Actor, normalize_state, get_patrol_positions


# Environment Parameters
GRID_SIZE = 1000
TIME_STEP = 0.1
NUM_ROBOTS = 4
NUM_TARGETS = 15
MAX_SPEED = 20
DETECTION_RADIUS = 15  # Adjust as necessary
KILL_RADIUS = 2  # Adjust as necessary
USE_TRANSFORMER = False  # Use transformer model if True
PATROL_RADIUS = 10
STATE_SIZE = 4 + (NUM_TARGETS * 2) + ((NUM_ROBOTS - 1) * 2) + 2  # Include central_obj's position
ACTION_SIZE = 5  # [up, down, left, right, stay]

ACTION_MAP = {
    0: (0, MAX_SPEED),    # Up
    1: (0, -MAX_SPEED),   # Down
    2: (-MAX_SPEED, 0),   # Left
    3: (MAX_SPEED, 0),    # Right
    4: (0, 0)             # Stay
}

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

class TransformerNN(torch.nn.Module):
    def __init__(self, state_dim, action_size, num_heads=4, num_layers=2):
        super(TransformerNN, self).__init__()
        self.state_dim = state_dim
        self.input_embedding = nn.Linear(state_dim, 64)
        self.positional_encoding = nn.Parameter(torch.zeros(1, state_dim, 64))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        x = self.input_embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Mean pooling
        return self.fc(x)

def load_model(model_path, state_size, action_size):
    model = TransformerNN(state_size, action_size) if USE_TRANSFORMER else UpdatedModel(state_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"{'TransformerNN' if USE_TRANSFORMER else 'SimpleNN'} model weights loaded successfully!")
    return model

def choose_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

def run_simulation(model):
    # Initialize central object
    central_waypoints = [(200, 200), (800, 200), (800, 800), (200, 800)]
    central_obj = CentralObject(central_waypoints[0][0], central_waypoints[0][1], max_speed=8, waypoints=central_waypoints)

    # Initialize robots in patrol formation
    robots = [Actor(px, py, MAX_SPEED) for (px, py) in get_patrol_positions(central_obj)]

    # Define waypoints for 15 targets
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

    # Set up visualization
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_title("Patrol & Protect Showcase")

    central_dot, = ax.plot([central_obj.x], [central_obj.y], 'r*', markersize=10, label='Central Object')
    colors = ['bo', 'go', 'ro', 'mo']
    robot_dots = [ax.plot([], [], c)[0] for c in colors]
    target_dots = [ax.plot([], [], 'yo')[0] for _ in range(NUM_TARGETS)]
    ax.legend()

    def init():
        for dot in robot_dots + target_dots:
            dot.set_data([], [])
        return robot_dots + target_dots + [central_dot]

    def update(frame):
        central_obj.update(random_walk=False)
        for t in targets:
            t.update()
        patrol_positions = get_patrol_positions(central_obj)
        for i, robot in enumerate(robots):
            state = normalize_state([
                robot.x, robot.y, robot.vx, robot.vy, central_obj.x, central_obj.y
            ] + [target.x - robot.x for target in targets] + [target.y - robot.y for target in targets], STATE_SIZE)
            action = choose_action(state, model)
            ax, ay = ACTION_MAP[action]
            robot.set_velocity(ax, ay)
            robot.update_position()
            robot_dots[i].set_data([robot.x], [robot.y])
        for i, t in enumerate(targets):
            target_dots[i].set_data([t.x], [t.y])
        central_dot.set_data([central_obj.x], [central_obj.y])
        return robot_dots + target_dots + [central_dot]

    ani = FuncAnimation(fig, update, init_func=init, frames=200, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    model_path = "/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/Patrol&Protect/multi_robot_patrol_model.pth"
    model = load_model(model_path, STATE_SIZE, ACTION_SIZE)
    run_simulation(model)