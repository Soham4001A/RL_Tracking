import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt
from Sharedutils import *


STATE_SIZE = 4 + 2 * (NUM_ROBOTS - 1)
ACTION_SIZE = 5  # [up, down, left, right, stay]



# Load trained model
def load_model(model_path, state_size, action_size):
    model = SimpleNN(state_size, action_size)
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model weights loaded successfully!")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Attempting to remap keys...")

        pretrained_state = torch.load(model_path)
        # Update keys to match current model
        new_state = {
            key.replace("fc0.", "fc.0.").replace("fc2.", "fc.2."): value
            for key, value in pretrained_state.items()
        }
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

            # Add distances to other robots to the state
            distances_to_others = []
            for j, other_robot in enumerate(robots):
                if i != j:
                    dist_x = other_robot.x - robot.x
                    dist_y = other_robot.y - robot.y
                    distances_to_others.extend([dist_x, dist_y])

            # Create the state vector with velocities and distances
            state = [dx, dy, robot.vx, robot.vy] + distances_to_others

            # Normalize the state
            state = [s / GRID_SIZE for s in state]

            # Choose the action
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

            # Update the robot's position
            robot.update_position()

            # Compute the distance and reward
            distance = sqrt(dx**2 + dy**2)
            _ = compute_reward(distance, prev_distances[i], i, robots)
            prev_distances[i] = distance

            # Update visualization for this robot
            robot_dots[i].set_data([robot.x], [robot.y])

        # Update target visualization
        target_dot.set_data([target.x], [target.y])
        return robot_dots + [target_dot]

    ani = FuncAnimation(fig, update, init_func=init, frames=200, interval=50, blit=True)
    plt.show()

# Main function
if __name__ == "__main__":
    model_path = "/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/MultiRobot/simple_rl_model_multi_robot.pth"
    model = load_model(model_path, STATE_SIZE, ACTION_SIZE)
    run_simulation(model)