from simulation import *
from classes import *


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

from classes import SimulationEngine, CCA, Foxtrot
from simulation import PPOEnv
from globals import *

# Visualization parameters
NUM_CCA = num_cca
STEPS = 100

class ShowcaseSimulation:
    def __init__(self, model_path, env, steps):
        self.model = PPO.load(model_path)
        self.env = env
        self.steps = steps
        self.cca_positions = []
        self.foxtrot_positions = []

    def run_simulation(self):
        """Run the simulation and collect positions for visualization."""
        obs = self.env.reset()
        for _ in range(self.steps):
            # Get the action from the trained model
            action, _ = self.model.predict(obs)

            # Pass action directly to the step method
            obs, reward, done, info = self.env.step(action)

            # Store positions for visualization
            self.cca_positions.append(np.array([np.array(pos) for pos in self.env.cca_positions], dtype=float))
            self.foxtrot_positions.append(np.array(self.env.foxtrot_position, dtype=float))

            if done:
                break

    def visualize(self):
        """Visualize the simulation results."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set up the grid
        ax.set_xlim([0, grid_size])
        ax.set_ylim([0, grid_size])
        ax.set_zlim([0, grid_size])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        cca_paths = [np.array([]).reshape(0, 3) for _ in range(NUM_CCA)]
        foxtrot_path = np.array([]).reshape(0, 3)

        # Update function for animation
        def update(frame):
            ax.cla()
            ax.set_xlim([0, grid_size])
            ax.set_ylim([0, grid_size])
            ax.set_zlim([0, grid_size])
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            for i in range(NUM_CCA):
                # Safeguard against insufficient or incorrect data
                if frame >= len(self.cca_positions) or len(self.cca_positions[0]) <= i:
                    continue

                # Extract the path for the current CCA
                cca_path = np.array([pos[i] for pos in self.cca_positions[:frame]])
                if len(cca_path) > 0:  # Ensure path has data
                    ax.plot(cca_path[:, 0], cca_path[:, 1], cca_path[:, 2], color='green', label=f'CCA {i + 1} Path')
                    ax.scatter(*cca_path[-1], color='green', s=100)

            if frame < len(self.foxtrot_positions):
                # Extract the path for Foxtrot
                foxtrot_path = np.array(self.foxtrot_positions[:frame])
                if len(foxtrot_path) > 0:  # Ensure path has data
                    ax.plot(foxtrot_path[:, 0], foxtrot_path[:, 1], foxtrot_path[:, 2], color='orange', label='Foxtrot Path')
                    ax.scatter(*foxtrot_path[-1], color='orange', s=100)

            ax.legend()

        # Create animation
        anim = FuncAnimation(fig, update, frames=self.steps, interval=200)
        plt.show()


if __name__ == "__main__":
    # Initialize the environment
    env = PPOEnv(grid_size=grid_size, num_cca=NUM_CCA)

    # Initialize the showcase simulation
    showcase = ShowcaseSimulation(model_path="./PPO_V2/Trained_Model", env=env, steps=STEPS)

    # Run the simulation
    showcase.run_simulation()

    # Visualize the results
    showcase.visualize()