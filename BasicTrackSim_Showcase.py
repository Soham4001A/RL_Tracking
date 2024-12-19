import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, cos, sin

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 20
STATE_SIZE = 4  # [dx, dy, robot_vx, robot_vy]
ACTION_SIZE = 4  # [up, down, left, right, stay]

# Actor class for robot and target
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

# Updated model to align with the previous architecture
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

# Load the trained model with compatibility adjustments
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

# Choose action using the updated model
def choose_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

# Simulation function
def run_simulation(model):
    # Initialize robot and target
    robot = Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED)
    target = Actor(np.random.uniform(0, GRID_SIZE), np.random.uniform(0, GRID_SIZE), MAX_SPEED)

    # Visualization setup
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    robot_dot, = ax.plot([], [], 'bo', label='Robot')
    target_dot, = ax.plot([], [], 'ro', label='Target')
    ax.legend()

    # Animation initialization
    def init():
        robot_dot.set_data([], [])
        target_dot.set_data([], [])
        return robot_dot, target_dot

    # Animation update function
    def update(frame):
        nonlocal robot, target

        # Move target along a fixed path
        target.update_fixed_path()

        # Calculate state
        dx, dy = target.x - robot.x, target.y - robot.y
        state = [dx, dy, robot.vx, robot.vy]

        # Choose action using the updated model
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

        # Update visualization
        robot_dot.set_data([robot.x], [robot.y])
        target_dot.set_data([target.x], [target.y])

        return robot_dot, target_dot

    # Create animation
    ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True)

    # Display animation
    print("Running simulation with the updated model. Close the window to finish.")
    plt.show()

# Main function
if __name__ == "__main__":
    model_path = "/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/simple_rl_model.pth"  # Updated model path
    model = load_model(model_path, STATE_SIZE, ACTION_SIZE)

    # Run the simulation
    run_simulation(model)