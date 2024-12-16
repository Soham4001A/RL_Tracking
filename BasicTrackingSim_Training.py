import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sqrt

VISUALIZATION = False

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 20
MIN_DISTANCE = sqrt(2)  # Distance threshold for penalization

# RL Parameters
STATE_SIZE = 4  # [dx, dy, vx, vy]
ACTION_SIZE = 5  # [up, down, left, right, stay]
GAMMA = 0.99  # Discount factor
LR = 0.001  # Learning rate
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000

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
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])
            q_values = self.model(state)
            next_q_values = self.target_model(next_state)
            target = reward + (1 - done) * self.gamma * torch.max(next_q_values)
            q_values[action] = target
            loss = torch.nn.functional.mse_loss(self.model(state), q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Environment Simulation
def compute_reward(distance):
    if distance < MIN_DISTANCE:
        return -10.0  # Penalty for being too close
    return -1.0 * distance  # Reward for getting closer but not too close

def simulate_episode(agent, robot, target, num_steps=1000):
    total_reward = 0
    for step in range(num_steps):
        dx, dy = target.x - robot.x, target.y - robot.y
        state = [dx, dy, robot.vx, robot.vy]
        action = agent.act(state)

        # Execute action
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
        target.set_velocity(np.random.uniform(-MAX_SPEED, MAX_SPEED), np.random.uniform(-MAX_SPEED, MAX_SPEED))
        target.update_position()

        # Calculate reward
        distance = np.sqrt(dx**2 + dy**2)
        reward = compute_reward(distance)

        # Observe next state
        next_dx, next_dy = target.x - robot.x, target.y - robot.y
        next_state = [next_dx, next_dy, robot.vx, robot.vy]
        done = False  # Define a termination condition if necessary
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward

        # Train the agent
        agent.replay()

    agent.update_target_model()
    return total_reward

# Visualization
def visualize(agent, robot, target, num_steps=200):
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    robot_dot, = ax.plot([], [], 'bo', label='Robot')
    target_dot, = ax.plot([], [], 'ro', label='Target')
    ax.legend()

    def init():
        robot_dot.set_data([], [])
        target_dot.set_data([], [])
        return robot_dot, target_dot

    def update(frame):
        robot.update_position()
        target.update_position()

        # Update the robot's position (must be a sequence)
        robot_dot.set_data([robot.x], [robot.y])  # Wrapping x and y in lists

        # Update the target's position (must also be a sequence)
        target_dot.set_data([target.x], [target.y])  # Wrapping x and y in lists

        return robot_dot, target_dot


    anim = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=50)
    plt.show()

# Main Training Loop
robot = Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED)
target = Actor(np.random.uniform(0, GRID_SIZE), np.random.uniform(0, GRID_SIZE), MAX_SPEED)
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

episodes = 100
for e in range(episodes):
    print(f"Episode {e + 1}/{episodes}, Training Agent...")
    reward = simulate_episode(agent, robot, target)
    print(f"Episode {e + 1}/{episodes}, Total Reward: {reward:.2f}")

    if VISUALIZATION: #Keep in mind that this is extremely intensive on the CPU/GPU
        
        # Reset and visualize the episode
        fig, ax = plt.subplots()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)

        robot_dot, = ax.plot([], [], 'bo', label='Robot')
        target_dot, = ax.plot([], [], 'ro', label='Target')
        ax.legend()

        def init():
            robot_dot.set_data([], [])
            target_dot.set_data([], [])
            return robot_dot, target_dot

        def update(frame):
            robot.update_position()
            target.update_position()

            robot_dot.set_data([robot.x], [robot.y])
            target_dot.set_data([target.x], [target.y])

            return robot_dot, target_dot

        ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True)

        # Display the figure and block until the user closes it
        print("Animation displayed. Close the figure window to proceed to the next episode.")
        plt.show(block=True)

        # Close the current figure before proceeding
        plt.close(fig)

# Save the trained model
torch.save(agent.model.state_dict(), "dqn_robot_model.pth")
print("Model saved as 'dqn_robot_model.pth'.")


