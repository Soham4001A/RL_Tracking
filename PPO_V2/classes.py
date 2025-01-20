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
    Movement function for Foxtrot to alternate between rectangle movement in x/y
    and pure z-axis movement to a random z-coordinate.

    Parameters:
        position (np.array): Current position of Foxtrot.
        cube_state (dict): Maintains state for the traversal, including phase and targets.

    Returns:
        new_position (np.array): Updated position of Foxtrot.
    """
    step_size = 10  # Step size for movement
    grid_size = 500  # Boundary of the grid

    # Initialize cube state if not already
    if "phase" not in cube_state:
        cube_state["phase"] = "rectangle"
        cube_state["rectangle"] = {
            "vertices": None,
            "current_index": 0,
            "steps_completed": 0,
        }
        cube_state["target_z"] = None

    # Phase 1: Rectangle Movement in x/y
    if cube_state["phase"] == "rectangle":
        rectangle = cube_state["rectangle"]

        # If rectangle is not initialized, create one with random length/width
        if rectangle["vertices"] is None:
            length = np.random.randint(50, 200)  # Random length
            width = np.random.randint(50, 200)  # Random width
            center = position[:2]  # Use current x, y as the center
            rectangle["vertices"] = [
                center + np.array([length / 2, width / 2]),
                center + np.array([-length / 2, width / 2]),
                center + np.array([-length / 2, -width / 2]),
                center + np.array([length / 2, -width / 2]),
            ]
            rectangle["current_index"] = 0
            rectangle["steps_completed"] = 0

        # Move along the rectangle vertices
        target_vertex = rectangle["vertices"][rectangle["current_index"]]
        direction = target_vertex - position[:2]
        distance = np.linalg.norm(direction)

        if distance <= step_size:
            # If close to the target vertex, move to it and switch to the next vertex
            position[:2] = target_vertex
            rectangle["current_index"] = (rectangle["current_index"] + 1) % len(rectangle["vertices"])
        else:
            # Move toward the target vertex
            direction = direction / distance  # Normalize
            position[:2] += (direction * step_size).astype(int)  # Explicitly cast to int

        rectangle["steps_completed"] += 1

        # After completing a full rectangle, switch to z-axis movement
        if rectangle["current_index"] == 0 and rectangle["steps_completed"] > 4:
            cube_state["phase"] = "z_movement"
            cube_state["target_z"] = np.random.randint(0, grid_size)  # Random target z

    # Phase 2: Pure z-axis Movement
    elif cube_state["phase"] == "z_movement":
        target_z = cube_state["target_z"]
        direction = target_z - position[2]

        if abs(direction) <= step_size:
            # If close to the target z, move to it and switch back to rectangle movement
            position[2] = target_z
            cube_state["phase"] = "rectangle"
            cube_state["rectangle"] = {"vertices": None, "current_index": 0, "steps_completed": 0}
        else:
            # Move along the z-axis
            position[2] += int(np.sign(direction) * step_size)  # Explicitly cast to int

    # Ensure position stays within grid boundaries
    position = np.clip(position, 0, grid_size)

    return np.round(position).astype(int)

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