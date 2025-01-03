# Environment Parameters
GRID_SIZE = 100
TIME_STEP = 0.1
MAX_SPEED = 20
NUM_ROBOTS = 4

import torch.nn as nn
import numpy as np

class Actor:
    def __init__(self, x, y, max_speed):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.max_speed = max_speed
        self.time = 0

    def update_fixed_path(self, side_length=30, center=(50, 50), speed=0.05):
        # Square path for the target
        perimeter = 4 * side_length
        self.time += speed * TIME_STEP
        distance_along_perimeter = (self.time * perimeter) % perimeter

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

    def set_velocity(self, ax, ay):
        self.vx = np.clip(ax, -self.max_speed, self.max_speed)
        self.vy = np.clip(ay, -self.max_speed, self.max_speed)

    def update_position(self):
        self.x += self.vx * TIME_STEP
        self.y += self.vy * TIME_STEP
        self.x = max(0, min(self.x, GRID_SIZE))
        self.y = max(0, min(self.y, GRID_SIZE))
    

class SimpleNN(nn.Module):
    def __init__(self, state_dim, action_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)