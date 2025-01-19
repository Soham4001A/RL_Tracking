import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Internal Module Imports
from globals import *

class BaseObject:
    """Base class for objects in the simulation."""
    def __init__(self, name, initial_position, color='blue'):
        self.name = name
        self.position = np.array(initial_position)
        self.prev_position = np.array(initial_position)
        self.path = []
        self.color = color

    def move(self, movement_fn):
        """Update the object's position using a movement function."""
        new_position = movement_fn(self.position)
        self.position = new_position
        self.path.append(new_position.copy())


class CCA(BaseObject):
    """Custom class for CCA objects."""
    def __init__(self, name, initial_position):
        super().__init__(name, initial_position, color='green')


class Foxtrot(BaseObject):
    """Custom class for Foxtrot objects."""
    def __init__(self, name, initial_position):
        super().__init__(name, initial_position, color='orange')


class SimulationEngine:
    def __init__(self, grid_size=1000):
        self.grid_size = grid_size
        self.objects = []
        self.fig = None
        self.ax = None

    def add_object(self, obj):
        """Add an object to the simulation."""
        if self.is_within_bounds(obj.position):
            self.objects.append(obj)
        else:
            print(f"Object {obj.name}'s initial position {obj.position} is out of bounds!")

    def is_within_bounds(self, position):
        """Check if the given position is within the grid boundaries."""
        return all(0 <= pos < self.grid_size for pos in position)

    def simulate(self, steps, movement_fns):
        """
        Simulate the objects' movements based on their movement functions.
        :param steps: Number of steps to simulate.
        :param movement_fns: Dictionary mapping object names to movement functions.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the grid
        self.ax.set_xlim([0, self.grid_size])
        self.ax.set_ylim([0, self.grid_size])
        self.ax.set_zlim([0, self.grid_size])
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')

        # Start the animation
        anim = FuncAnimation(self.fig, self.update_frame, frames=steps, fargs=(movement_fns,), interval=100)
        plt.show()

    def update_frame(self, frame, movement_fns):
        """Update the frame for the animation."""
        self.ax.cla()
        self.ax.set_xlim([0, self.grid_size])
        self.ax.set_ylim([0, self.grid_size])
        self.ax.set_zlim([0, self.grid_size])
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')

        # Update and render each object
        for obj in self.objects:
            movement_fn = movement_fns.get(obj.name, lambda pos: pos)
            obj.move(movement_fn)
            if not self.is_within_bounds(obj.position):
                print(f"Object {obj.name} moved out of bounds to {obj.position}!")
            else:
                # Plot the object's path and position
                if obj.path:
                    path = np.array(obj.path)
                    self.ax.plot(path[:, 0], path[:, 1], path[:, 2], label=f'{obj.name} Path', color=obj.color)
                self.ax.scatter(*obj.position, color=obj.color, label=obj.name, s=100)

        self.ax.legend()


# Movement Functions
def cca_movement_fn(position):
    """Movement function for CCA objects -> Basic"""
    return position + np.array([10, 0, 10])


def foxtrot_movement_fn(position):
    """Generate random movement for Foxtrot with dynamic edge handling."""
    step_size = 10  # Define step size
    direction_probabilities = [1, 1, 1]  # Weights for [X, Y, Z] movement
    
    # Randomly choose which axes to move on
    axes_to_move = np.random.choice([0, 1, 2], size=3, replace=False, p=np.array(direction_probabilities) / sum(direction_probabilities))

    # Generate random movement for each selected axis
    random_step = np.zeros(3, dtype=int)
    for axis in axes_to_move:
        random_step[axis] = np.random.choice([-step_size, step_size])

    # Update position
    new_position = position + random_step

    # Handle edges by bouncing back
    for i in range(3):
        if new_position[i] < 0:
            new_position[i] = abs(new_position[i])  # Reflect back
        elif new_position[i] > grid_size:
            new_position[i] = grid_size - (new_position[i] - grid_size)  # Reflect back from upper edge

    return new_position


def foxtrot_movement_fn_cube(position, cube_state):
    """
    Movement function for Foxtrot to follow a 3D cube pattern.

    Parameters:
        position (np.array): Current position of Foxtrot.
        cube_state (dict): Maintains state for the cube traversal, including current edge and progress.

    Returns:
        new_position (np.array): Updated position of Foxtrot.
    """
    side_length = 200  # Length of each side of the cube
    half_side = side_length // 2
    center = np.array([250, 250, 250])  # Center of the cube (adjusted for grid_size=500)

    # Cube vertices
    cube_vertices = [
        center + np.array([half_side, half_side, half_side]),
        center + np.array([-half_side, half_side, half_side]),
        center + np.array([-half_side, -half_side, half_side]),
        center + np.array([half_side, -half_side, half_side]),
        center + np.array([half_side, -half_side, -half_side]),
        center + np.array([half_side, half_side, -half_side]),
        center + np.array([-half_side, half_side, -half_side]),
        center + np.array([-half_side, -half_side, -half_side]),
    ]

    # Cube edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
        (0, 5), (1, 6), (2, 7), (3, 4)   # Vertical edges
    ]

    # Initialize cube state if not already
    if "current_edge" not in cube_state or "progress" not in cube_state:
        cube_state["current_edge"] = 0
        cube_state["progress"] = 0

    # Get the current edge and its vertices
    edge_index = cube_state["current_edge"]
    edge_start, edge_end = edges[edge_index]
    edge_progress = cube_state["progress"] / side_length

    # Interpolate position along the current edge
    new_position = (1 - edge_progress) * cube_vertices[edge_start] + edge_progress * cube_vertices[edge_end]

    # Update progress and edge
    cube_state["progress"] += 10  # Step size along the edge
    if cube_state["progress"] >= side_length:
        cube_state["progress"] = 0  # Reset progress
        cube_state["current_edge"] = (cube_state["current_edge"] + 1) % len(edges)  # Move to the next edge

    # Return the new position as an integer
    return np.round(new_position).astype(int)

class Transformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=64, num_heads=3, ff_hidden=128, num_layers=4, seq_len=6):
        feature_dim = embed_dim * seq_len
        super(Transformer, self).__init__(
            observation_space, features_dim=feature_dim
        )
        self.embed_dim = embed_dim
        self.input_dim = observation_space.shape[0]
        self.seq_len = seq_len  # Define seq_len explicitly
        
        # Validate that seq_len and embed_dim are compatible
        if self.input_dim % seq_len != 0:
            raise ValueError("Input dimension must be divisible by seq_len.")
        
        # Linear projection for input embedding
        self.input_embedding = nn.Linear(self.input_dim // seq_len, embed_dim)
        
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
        # Reshape input into (batch_size, seq_len, features_per_seq)
        batch_size = x.shape[0]
        features_per_seq = self.input_dim // self.seq_len
        x = x.view(batch_size, self.seq_len, features_per_seq)
        
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