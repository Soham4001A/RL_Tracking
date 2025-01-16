import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

#Internal Imports
from classes import *


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


# Example usage
def cca_movement_fn(position):
    """Movement function for CCA objects."""
    return position + np.array([10, 0, 10])


def foxtrot_movement_fn(position):
    """Movement function for Foxtrot objects to move in a random pattern."""
    step_size = 10  # Maximum step size in any direction
    random_step = np.random.randint(-step_size, step_size + 1, size=3)  # Random step for X, Y, Z
    return position + random_step
    

if __name__ == "__main__":
    engine = SimulationEngine(grid_size=1000)

    # Add objects to the simulation
    cca = CCA(name="CCA1", initial_position=[100, 100, 100])
    foxtrot = Foxtrot(name="Foxtrot1", initial_position=[200, 200, 200])
    engine.add_object(cca)
    engine.add_object(foxtrot)

    # Simulate with specific movement functions
    movement_functions = {
        "CCA1": cca_movement_fn,
        "Foxtrot1": foxtrot_movement_fn
    }
    engine.simulate(steps=50, movement_fns=movement_functions)