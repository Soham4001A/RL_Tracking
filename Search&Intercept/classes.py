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
    Movement function for Foxtrot to alternate between:
      - Rectangle movement in (x,y) while z is fixed
      - Pure z-axis movement to a random z-coordinate
    Repeats these two phases infinitely.
    """

    # Make sure we're working in float space so partial steps won't error
    position = position.astype(float)

    if "phase" not in cube_state:
        cube_state["phase"] = "rectangle"
        cube_state["rectangle"] = {
            "vertices": None,
            "current_index": 0,
            "laps_completed": 0,
        }
        cube_state["target_z"] = None

    if cube_state["phase"] == "rectangle":
        rect_info = cube_state["rectangle"]

        if rect_info["vertices"] is None:
            # Pick random rectangle size
            length = np.random.randint(50, 200)
            width = np.random.randint(50, 200)

            center = position[:2]  # current (x, y)
            v0 = center + np.array([+length/2, +width/2])
            v1 = center + np.array([-length/2, +width/2])
            v2 = center + np.array([-length/2, -width/2])
            v3 = center + np.array([+length/2, -width/2])

            rect_info["vertices"] = [v0, v1, v2, v3]
            rect_info["current_index"] = 0
            rect_info["laps_completed"] = 0

        vertices = rect_info["vertices"]
        target_vertex = vertices[rect_info["current_index"]]

        direction_2d = target_vertex - position[:2]
        dist_2d = np.linalg.norm(direction_2d)

        if dist_2d <= step_size:
            # Snap to corner
            position[:2] = target_vertex
            # Advance
            rect_info["current_index"] = (rect_info["current_index"] + 1) % len(vertices)

            # If we looped back to corner 0, one lap is done
            if rect_info["current_index"] == 0:
                rect_info["laps_completed"] += 1
                # Switch to z phase after 1 complete loop
                if rect_info["laps_completed"] >= 1:
                    cube_state["phase"] = "z_movement"
                    cube_state["target_z"] = np.random.randint(0, grid_size)

        else:
            # Take a partial step
            step_dir = direction_2d / dist_2d  # unit vector
            position[:2] += step_dir * step_size

    elif cube_state["phase"] == "z_movement":
        target_z = cube_state["target_z"]
        dz = target_z - position[2]

        if abs(dz) <= step_size:
            # Snap to final z
            position[2] = target_z
            # Go back to rectangle
            cube_state["phase"] = "rectangle"
            # Force a fresh rectangle next time
            cube_state["rectangle"] = {
                "vertices": None,
                "current_index": 0,
                "laps_completed": 0,
            }
        else:
            # Move up or down
            position[2] += np.sign(dz) * step_size

    # Clip, round, cast back to int so environment remains consistent
    position = np.clip(position, 0, grid_size - 1)
    position = np.round(position).astype(int)
    return position

class GridSpace:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # Create a 3D grid where each cell is initially 0
        self.grid = np.full((grid_size, grid_size, grid_size), 0, dtype=object)

    def update_robot_area(self, prev_center, new_center, robot_name, observable_radius):
        half = observable_radius // 2
        # Clear previous area: remove robot string and increment the cell's integer
        for x in range(max(0, prev_center[0] - half), min(self.grid_size, prev_center[0] + half + 1)):
            for y in range(max(0, prev_center[1] - half), min(self.grid_size, prev_center[1] + half + 1)):
                for z in range(max(0, prev_center[2] - half), min(self.grid_size, prev_center[2] + half + 1)):
                    cell = self.grid[x, y, z]
                    if isinstance(cell, tuple) and cell[1] == robot_name:
                        # Remove the string by setting the cell back to an integer (incremented by 1)
                        self.grid[x, y, z] = cell[0] + 1
        
        # Set new area: mark cells within the observable cube with a tuple (current integer, robot_name)
        for x in range(max(0, new_center[0] - half), min(self.grid_size, new_center[0] + half + 1)):
            for y in range(max(0, new_center[1] - half), min(self.grid_size, new_center[1] + half + 1)):
                for z in range(max(0, new_center[2] - half), min(self.grid_size, new_center[2] + half + 1)):
                    cell = self.grid[x, y, z]
                    if isinstance(cell, int):
                        self.grid[x, y, z] = (cell, robot_name)

    def update_target_area(self, prev_pos, new_pos):
        x, y, z = prev_pos
        cell = self.grid[x, y, z]
        if isinstance(cell, tuple) and cell[1] == "target":
            # Remove the target string and increment the cell's integer
            self.grid[x, y, z] = cell[0] + 1
        
        x, y, z = new_pos
        cell = self.grid[x, y, z]
        if isinstance(cell, int):
            self.grid[x, y, z] = (cell, "target")


class Transformer(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        embed_dim=64,
        num_heads=3,
        ff_hidden=128,
        num_layers=4,
        seq_len=6,
        dropout=0.3
    ):
        """
        :param observation_space: Gym observation space.
        :param embed_dim: Size of the embedding (d_model) in the Transformer.
        :param num_heads: Number of attention heads in the multi-head attention layers.
        :param ff_hidden: Dimension of the feedforward network in the Transformer.
        :param num_layers: Number of layers in the Transformer encoder.
        :param seq_len: Number of time steps to unroll in the Transformer.
        :param dropout: Dropout probability to use throughout the model.
        """
        # Features dimension after Transformer = embed_dim * seq_len
        feature_dim = embed_dim * seq_len
        super(Transformer, self).__init__(observation_space, features_dim=feature_dim)

        self.embed_dim = embed_dim
        self.input_dim = observation_space.shape[0]
        self.seq_len = seq_len  
        self.dropout_p = dropout

        # Validate that seq_len divides input_dim evenly
        if self.input_dim % seq_len != 0:
            raise ValueError("Input dimension must be divisible by seq_len.")

        # Linear projection for input -> embedding
        self.input_embedding = nn.Linear(self.input_dim // seq_len, embed_dim)

        # Dropout layer for embeddings
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)

        # Transformer Encoder
        #   - The 'dropout' parameter here applies to:
        #       1) The self-attention mechanism outputs
        #       2) The feed-forward sub-layer outputs
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=self.dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Flatten final output to feed into the policy & value networks
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Reshape input into (batch_size, seq_len, features_per_seq)
        batch_size = x.shape[0]
        features_per_seq = self.input_dim // self.seq_len
        x = x.view(batch_size, self.seq_len, features_per_seq)

        # Linear projection
        x = self.input_embedding(x)

        # Add positional encoding
        batch_size, seq_len, embed_dim = x.shape
        x = x + self._positional_encoding(seq_len, embed_dim).to(x.device)

        # Drop out some embeddings for regularization
        x = self.embedding_dropout(x)

        # Pass sequence through the Transformer encoder
        x = self.transformer(x)

        # Flatten for final feature vector
        return self.flatten(x)

    def _positional_encoding(self, seq_len, embed_dim):
        """Sine-cosine positional encoding, shape: (seq_len, embed_dim)."""
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # The standard “div_term” for sine/cosine in attention
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    