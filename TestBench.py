import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt, cos, sin, pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 10
MIN_DISTANCE = sqrt(2)
VISUALIZATION = False

# RL Parameters
SEQ_LENGTH = 10  # Number of past states to feed into the transformer
STATE_DIM = 4   # Single state dimension: [dx, dy, vx, vy]
STATE_SIZE = STATE_DIM  # single step state size is still 4
ACTION_SIZE = 5  # [up, down, left, right, stay]
GAMMA = 0.75
LR = 0.0003  # PPO learning rate
CLIP_EPS = 0.25  # PPO clipping epsilon
EPOCHS = 25
BATCH_SIZE = 32

ACTION_MAP = {
    0: (0, MAX_SPEED),   # Up
    1: (0, -MAX_SPEED),  # Down
    2: (-MAX_SPEED, 0),  # Left
    3: (MAX_SPEED, 0),   # Right
    4: (0, 0)            # Stay
}

# Actor class for robot and target
class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed
        self.time = 0

    def update_fixed_path(self, radius=30, center=(50, 50), speed=0.05):
        # Circular motion path
        self.time += speed
        self.x = center[0] + radius * cos(2 * pi * self.time)
        self.y = center[1] + radius * sin(2 * pi * self.time)

    def set_velocity(self, vx, vy):
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = max(0, min(self.x, GRID_SIZE))
        self.y = max(0, min(self.y, GRID_SIZE))

# Transformer Network
class TransformerNetwork(nn.Module):
    def __init__(self, state_size, action_size, d_model=64, nhead=2, dim_feedforward=128, num_layers=2):
        super(TransformerNetwork, self).__init__()
        self.input_layer = nn.Linear(state_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers,
        )
        self.policy_head = nn.Linear(d_model, action_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, state_size)
        batch_size, seq_length, _ = x.shape
        x = self.input_layer(x)  # (batch, seq, d_model)
        # Transformer expects (seq, batch, d_model)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.transformer(x)  # (seq, batch, d_model)
        x = x.transpose(0, 1)  # (batch, seq, d_model)

        # Take the output of the last time step for policy/value
        x_last = x[:, -1, :]  # (batch, d_model)
        policy = self.policy_head(x_last)
        value = self.value_head(x_last)
        return policy, value

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_size):
        self.policy_network = TransformerNetwork(state_dim, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)
        self.action_size = action_size
        self.memory = []

    def remember(self, state_seq, action, reward, next_state_seq, done):
        # state_seq and next_state_seq are (SEQ_LENGTH, STATE_DIM)
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def act(self, state_seq):
        # state_seq shape: (SEQ_LENGTH, STATE_DIM)
        state_tensor = torch.tensor(state_seq, dtype=torch.float).unsqueeze(0)  # (1, seq, state_dim)
        with torch.no_grad():
            logits, _ = self.policy_network(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action

    def learn(self):
        # If no memory collected, return
        if len(self.memory) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float)        # (N, SEQ_LENGTH, STATE_DIM)
        actions = torch.tensor(actions, dtype=torch.long)       # (N,)
        rewards = torch.tensor(rewards, dtype=torch.float)      # (N,)
        next_states = torch.tensor(next_states, dtype=torch.float) # (N, SEQ_LENGTH, STATE_DIM)
        dones = torch.tensor(dones, dtype=torch.bool)           # (N,)

        # Compute old values and old probabilities
        with torch.no_grad():
            old_logits, old_values = self.policy_network(states)
            old_probs = torch.softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute discounted returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float)

        # Compute advantages
        with torch.no_grad():
            _, values = self.policy_network(states)
            values = values.squeeze()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(EPOCHS):
            logits, value_estimates = self.policy_network(states)
            value_estimates = value_estimates.squeeze()
            probs = torch.softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze()

            ratio = probs / (old_probs + 1e-8)
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

            # Entropy bonus for exploration
            full_probs = torch.softmax(logits, dim=-1)
            entropy = -(full_probs * torch.log(full_probs + 1e-8)).sum(-1).mean()

            actor_loss = -torch.min(ratio * advantages, clipped).mean()
            critic_loss = 0.5 * nn.functional.mse_loss(value_estimates, returns)
            loss = actor_loss + critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

def compute_reward(distance, robot, target, prev_vx, prev_vy):
    # A smoother, more incremental reward structure

    # Base small negative reward for distance
    reward = float(0)
    #reward = -0.1 * distance

    # Base distance reward function
    if distance > MIN_DISTANCE:
        if distance != 15:
            reward += 15-(distance/2)
        else:
            reward += 15.0001-(distance/2)
    else:
    # If too close (below MIN_DISTANCE), add a penalty
        reward -= 50.0

    if distance < sqrt(9):
        reward += 50
        
    # Encourage movement toward the target direction
    #direction_factor = ((target.x - robot.x)*robot.vx + (target.y - robot.y)*robot.vy) / (distance + 1e-8)
    #reward += 0.1 * direction_factor

    # Penalize oscillations in velocity (changes from previous step)
    #vel_change = abs(robot.vx - prev_vx) + abs(robot.vy - prev_vy)
    #reward -= 0.1 * vel_change

    # Penalize standing still (mildly)
    #if robot.vx == 0 and robot.vy == 0:
    #    reward -= 1.0

    print(f"Reward is {reward}")
    #print(f"Distance is {distance}")
    return reward

def run_simulation(agent, robot, target, num_steps=200):
    steps_counter = [0]  # Track steps per episode

    def reset_positions():
        robot.x, robot.y = GRID_SIZE / 2, GRID_SIZE / 2
        robot.vx, robot.vy = 0, 0
        target.x, target.y = 50, 50
        target.time = 0

    reset_positions()

    # For visualization
    fig = None
    anim = None
    if VISUALIZATION:
        fig, ax = plt.subplots()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        robot_dot, = ax.plot([], [], 'bo', label="Robot")
        target_dot, = ax.plot([], [], 'ro', label="Target")
        ax.legend()
        steps_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='blue')

    prev_vx, prev_vy = robot.vx, robot.vy

    # Initialize a queue of states
    initial_state = [target.x - robot.x, target.y - robot.y, robot.vx, robot.vy]
    state_buffer = deque([initial_state for _ in range(SEQ_LENGTH)], maxlen=SEQ_LENGTH)

    def step_function(agent, robot, target, state_buffer, steps_counter, num_steps):
        """
        Executes one step in the simulation, updates the robot's state,
        computes reward, and adds to agent memory.
        """
        # Increment step counter
        steps_counter[0] += 1

        # End the episode if max steps reached
        if steps_counter[0] >= num_steps:
            return None

        # Update target's position (fixed path)
        target.update_fixed_path()

        # Compute the relative state and distance
        dx, dy = target.x - robot.x, target.y - robot.y
        distance = sqrt(dx**2 + dy**2)

        # Append current state to the buffer
        current_state = [dx, dy, robot.vx, robot.vy]
        state_buffer.append(current_state)
        state_seq = np.array(state_buffer)

        # Choose action using the PPO agent
        action = agent.act(state_seq)
        vx, vy = ACTION_MAP[action]
        robot.set_velocity(vx, vy)

        # Update robot's position
        robot.update_position()

        # Compute reward
        reward = compute_reward(distance, robot, target, prev_vx, prev_vy)

        # Observe next state and add to memory
        next_dx, next_dy = target.x - robot.x, target.y - robot.y
        next_state = [next_dx, next_dy, robot.vx, robot.vy]
        state_buffer.append(next_state)
        next_state_seq = np.array(state_buffer)

        done = steps_counter[0] >= num_steps
        agent.remember(state_seq, action, reward, next_state_seq, done)

        return robot.x, robot.y, target.x, target.y, steps_counter[0], done

    def init():
        robot_dot.set_data([], [])
        target_dot.set_data([], [])
        steps_text.set_text("Steps: 0")
        return robot_dot, target_dot, steps_text

    def update(frame):
        result = step_function(agent, robot, target, state_buffer, steps_counter, num_steps)
        if result is None:
            print(f"Episode complete. Total steps: {steps_counter[0]}")
            anim.event_source.stop()
            plt.close(fig)
            return

        rx, ry, tx, ty, steps, done = result
        robot_dot.set_data([rx], [ry])
        target_dot.set_data([tx], [ty])
        steps_text.set_text(f"Steps: {steps}")
        return robot_dot, target_dot, steps_text

    if VISUALIZATION:
        anim = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=50)
        plt.show()
    else:
        for _ in range(num_steps):
            res = step_function(agent, robot, target, state_buffer, steps_counter, num_steps)
            if res is None:
                break

    agent.learn()

# Main Training Loop
robot = Actor(GRID_SIZE / 2, GRID_SIZE / 2, MAX_SPEED)
target = Actor(50, 50, MAX_SPEED)
agent = PPOAgent(STATE_SIZE, ACTION_SIZE)

episodes = 250
for episode in range(episodes):
    print(f"Episode {episode + 1}/{episodes}")
    run_simulation(agent, robot, target)

torch.save(agent.policy_network.state_dict(), "ppo_transformer_model.pth")
print("Model saved successfully!")