import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 20
STATE_SIZE = 4  # [dx, dy, vx, vy]
ACTION_SIZE = 5  # [up, down, left, right, stay]

# Actor class for robot and target
class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = max(0, min(self.x, GRID_SIZE))
        self.y = max(0, min(self.y, GRID_SIZE))

    def set_velocity(self, vx, vy):
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

# Q-Network for DQN
class QNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load the trained model
def load_model(model_path, state_size, action_size):
    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Explicitly set weights_only=True
    model.eval()  # Set the model to evaluation mode
    print("Model weights loaded successfully!")
    return model


# Choose action using the loaded model
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

        # Move target randomly
        target.set_velocity(np.random.uniform(-MAX_SPEED, MAX_SPEED),
                            np.random.uniform(-MAX_SPEED, MAX_SPEED))
        target.update_position()

        # Calculate state
        dx, dy = target.x - robot.x, target.y - robot.y
        state = [dx, dy, robot.vx, robot.vy]

        # Choose action using the loaded model
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
        robot_dot.set_data([robot.x], [robot.y])  # Wrapping robot.x and robot.y in lists.
        target_dot.set_data([target.x], [target.y])

        return robot_dot, target_dot

    # Create animation
    ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True)

    # Display animation
    print("Running simulation. Close the window to finish.")
    plt.show()

# Main function
if __name__ == "__main__":
    model_path = "/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/dqn_robot_model.pth"  # Path to the saved model
    model = load_model(model_path, STATE_SIZE, ACTION_SIZE)

    # Run the simulation
    run_simulation(model)
