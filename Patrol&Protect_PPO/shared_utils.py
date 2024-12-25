import numpy as np
import gymnasium as gym  # <-- Use Gymnasium instead of legacy gym
from math import sqrt, cos, sin, pi
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution

# Constants
GRID_SIZE = 1000
TIME_STEP = 0.1

# Patrol Radius
PATROL_RADIUS = sqrt((3)**2 + (3)**2)

# Action Maps
ACTION_MAP = {
    0: (0, 10),    # Up
    1: (0, -10),   # Down
    2: (-10, 0),   # Left
    3: (10, 0),    # Right
    4: (0, 0)      # Stay
}

# ----------------------------------------
# Classes for the Simulation Entities
# ----------------------------------------

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
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.x, self.y = self.waypoints[self.current_waypoint_idx]
        self.max_speed = max_speed

    def move_to_next_waypoint(self):
        """Move toward the current waypoint."""
        target_x, target_y = self.waypoints[self.current_waypoint_idx]
        direction_x = target_x - self.x
        direction_y = target_y - self.y
        distance = sqrt(direction_x**2 + direction_y**2)

        if distance < 0.1:  # threshold to consider the waypoint reached
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            target_x, target_y = self.waypoints[self.current_waypoint_idx]
            direction_x = target_x - self.x
            direction_y = target_y - self.y
            distance = sqrt(direction_x**2 + direction_y**2)

        if distance > 0:
            norm_x = direction_x / distance
            norm_y = direction_y / distance
            self.x += norm_x * self.max_speed * TIME_STEP
            self.y += norm_y * self.max_speed * TIME_STEP

    def update(self):
        """Update the target's position."""
        self.move_to_next_waypoint()


class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed

        # Store the most recently chosen action index.
        self.chosen_action = None

    def set_action(self, action):
        """
        Set the action index (0..4) and update velocity accordingly.
        Also store the chosen_action for debugging or record-keeping.
        """
        self.chosen_action = action
        vx, vy = self._action_to_velocity(action)
        self.set_velocity(vx, vy)

    def _action_to_velocity(self, a):
        """
        Convert integer action 'a' to a velocity (vx, vy).
        """
        if a == 0:
            velocity = (0, self.max_speed)    # Up
        elif a == 1:
            velocity = (0, -self.max_speed)  # Down
        elif a == 2:
            velocity = (-self.max_speed, 0)  # Left
        elif a == 3:
            velocity = (self.max_speed, 0)   # Right
        else:  # 4 => stay
            velocity = (0, 0)
        return velocity

    def set_velocity(self, vx, vy):
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = np.clip(self.x, 0, GRID_SIZE)
        self.y = np.clip(self.y, 0, GRID_SIZE)
# ----------------------------------------
# Utility Functions
# ----------------------------------------

def normalize_state(vals, state_dim, grid_size=GRID_SIZE, max_speed=10):
    """Normalize the state vector."""
    normalized = []
    for i, v in enumerate(vals):
        if i < 2:  # Positions
            normalized.append(v / grid_size)
        elif i < 4:  # Velocities
            normalized.append(v / max_speed)
        else:  # Relative positions
            normalized.append(v / grid_size)

    # Pad with zeros to match state_dim if needed
    while len(normalized) < state_dim:
        normalized.append(0.0)

    return normalized

def get_patrol_positions(central_obj, patrol_radius=PATROL_RADIUS):
    """Get patrol positions around the central object."""
    return [
        (central_obj.x + patrol_radius, central_obj.y),
        (central_obj.x - patrol_radius, central_obj.y),
        (central_obj.x, central_obj.y + patrol_radius),
        (central_obj.x, central_obj.y - patrol_radius)
    ]




class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Transformer Feature Extractor with 3 heads.
    """
    def __init__(self, observation_space, embed_dim=64, num_heads=3, ff_hidden=128, num_layers=2):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=embed_dim)
        self.embed_dim = embed_dim
        self.input_dim = observation_space.shape[0]
        
        # Linear projection for input embedding
        self.input_embedding = nn.Linear(self.input_dim, embed_dim)
        
        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_hidden, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Flatten final output
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        # Project input to embedding space
        x = self.input_embedding(x)
        
        # Add positional encoding (optional)
        batch_size, seq_len, embed_dim = x.shape
        x = x + self._positional_encoding(seq_len, embed_dim).to(x.device)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Flatten output
        return self.flatten(x)
    
    def _positional_encoding(self, seq_len, embed_dim):
        """
        Generate positional encoding.
        """
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe