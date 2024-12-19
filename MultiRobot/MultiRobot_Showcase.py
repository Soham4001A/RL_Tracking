import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 20
STATE_SIZE = 4  # [dx, dy, robot_vx, robot_vy]
ACTION_SIZE = 5  # [up, down, left, right, stay]
NUM_ROBOTS = 4

# Actor class for robots and target
class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed
        self.time = 0

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = max(0, min(self.x, GRID_SIZE))
        self.y = max(0, min(self.y, GRID_SIZE))

    def update_fixed_path(self, side_length=30, center=(50, 50), speed=0.05):
        # Calculate the total perimeter of the square
        perimeter = 4 * side_length
        # Calculate the current position along the perimeter based on time
        self.time += speed * TIME_STEP
        distance_along_perimeter = (self.time * perimeter) % perimeter

        # Determine which side of the square the target is on
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

    def set_velocity(self, vx, vy):
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

# Neural network model
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

# Load trained model
def load_model(model_path, state_size, action_size):
    model = UpdatedModel(state_size, action_size)
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model weights loaded successfully!")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Attempting to remap keys...")
        pretrained_state = torch.load(model_path)
        new_state = {key.replace("fc.", "fc"): value for key, value in pretrained_state.items()}
        model.load_state_dict(new_state)
        print("Remapped model weights loaded successfully!")
    model.eval()
    return model

def choose_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

def compute_reward(distance, prev_distance, robot_idx, robots):
    # Base reward: closer distance is better
    reward = -distance / GRID_SIZE
    if distance < prev_distance:
        reward += 0.5

    # Collision penalty
    for j, other_robot in enumerate(robots):
        if j != robot_idx:
            dist_to_other = sqrt((robots[robot_idx].x - other_robot.x)**2 + 
                                 (robots[robot_idx].y - other_robot.y)**2)
            if dist_to_other < 2:  # Collision threshold
                reward -= 1.0
                break
    return reward

# Simulation function
def run_simulation(model):
    # Initialize multiple robots at different starting positions
    start_x, start_y = GRID_SIZE / 2, GRID_SIZE / 2
    robots = [
        Actor(start_x, start_y, MAX_SPEED),
        Actor(start_x + 5, start_y, MAX_SPEED),
        Actor(start_x, start_y + 5, MAX_SPEED),
        Actor(start_x + 5, start_y + 5, MAX_SPEED)
    ]
    target = Actor(np.random.uniform(0, GRID_SIZE), np.random.uniform(0, GRID_SIZE), MAX_SPEED)
    
    # Track previous distances for reward calculation
    prev_distances = [float('inf')] * NUM_ROBOTS

    # Visualization setup
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    # Assign different colors/markers to each robot for clarity
    colors = ['bo', 'go', 'ro', 'mo']
    robot_dots = []
    for c in colors:
        dot, = ax.plot([], [], c)
        robot_dots.append(dot)
    target_dot, = ax.plot([], [], 'yo', label='Target')
    ax.legend()

    def init():
        for robot_dot in robot_dots:
            robot_dot.set_data([], [])
        target_dot.set_data([], [])
        return robot_dots + [target_dot]

    def update(frame):
        nonlocal robots, target, prev_distances

        # Move target along a fixed path
        target.update_fixed_path()

        # Update each robot
        for i, robot in enumerate(robots):
            dx, dy = target.x - robot.x, target.y - robot.y
            state = [dx, dy, robot.vx, robot.vy]
            action = choose_action(state, model)

            # Update robot velocity based on action
            if action == 0:  # up
                robot.set_velocity(0, MAX_SPEED)
            elif action == 1:  # down
                robot.set_velocity(0, -MAX_SPEED)
            elif action == 2:  # left
                robot.set_velocity(-MAX_SPEED, 0)
            elif action == 3:  # right
                robot.set_velocity(MAX_SPEED, 0)
            else:  # stay
                robot.set_velocity(0, 0)

            robot.update_position()
            distance = sqrt(dx**2 + dy**2)
            _ = compute_reward(distance, prev_distances[i], i, robots)
            prev_distances[i] = distance

            # Update visualization for this robot
            robot_dots[i].set_data([robot.x], [robot.y])

        # Update target visualization
        target_dot.set_data([target.x], [target.y])
        return robot_dots + [target_dot]

    # Note: Removed blit=True to avoid issues with multiple artists
    ani = FuncAnimation(fig, update, frames=200, init_func=init)

    print("Running simulation with multiple robots. Close the window to finish.")
    plt.show()

# Main function
if __name__ == "__main__":
    model_path = "/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/MultiRobot/simple_rl_model_multi_robot.pth"
    model = load_model(model_path, STATE_SIZE, ACTION_SIZE)
    run_simulation(model)