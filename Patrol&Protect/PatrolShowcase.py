import torch
import numpy as np
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

def load_model(model_path, state_size, action_size):
    model = UpdatedModel(state_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model weights loaded successfully!")
    return model

def choose_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

def normalize_state(vals):
    normalized = []
    for v in vals:
        normalized.append(v / GRID_SIZE)
    return normalized

def run_simulation(model):
    # Create a central object
    central_obj = CentralObject(CENTRAL_X, CENTRAL_Y)

    # Initialize robots in patrol formation
    robots = [Actor(px, py, MAX_SPEED) for (px, py) in PATROL_POSITIONS]

    # Initialize adversarial targets
    targets = [AdversarialTarget(np.random.uniform(0, GRID_SIZE), np.random.uniform(0, GRID_SIZE), MAX_SPEED) 
               for _ in range(NUM_TARGETS)]

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
        # Update targets
        for t in targets:
            t.update_random()

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
            state = normalize_state(state)

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