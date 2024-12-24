import numpy as np
from math import sqrt, cos, sin, pi
import torch
from torch import nn

# Constants
GRID_SIZE = 1000
TIME_STEP = 0.1

# Action Maps
ACTION_MAP = {
    0: (0, 10),    # Up
    1: (0, -10),   # Down
    2: (-10, 0),   # Left
    3: (10, 0),    # Right
    4: (0, 0)      # Stay
}

# Central Object Class
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

        # Update position
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


# Adversarial Target Class
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

        if distance < 0.1:  # Small threshold to consider the waypoint "reached"
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


# Actor Class
class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed

    def set_velocity(self, vx, vy):
        """Set the velocity of the robot."""
        self.vx = np.clip(vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(vy, -self.max_speed, self.max_speed)

    def update_position(self):
        """Update the position of the robot."""
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = np.clip(self.x, 0, GRID_SIZE)
        self.y = np.clip(self.y, 0, GRID_SIZE)


# Utility Functions
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

    # Pad with zeros to match state_dim
    while len(normalized) < state_dim:
        normalized.append(0.0)

    return normalized


def get_patrol_positions(central_obj, patrol_radius):
    """Get patrol positions around the central object."""
    return [
        (central_obj.x + patrol_radius, central_obj.y),
        (central_obj.x - patrol_radius, central_obj.y),
        (central_obj.x, central_obj.y + patrol_radius),
        (central_obj.x, central_obj.y - patrol_radius)
    ]