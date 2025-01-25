from simulation import PPOEnv
from classes import Transformer
import globals
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Visualization parameters
NUM_CCA = globals.num_cca
STEPS = 10000

class ShowcaseSimulation:
    def __init__(self, model_path, vec_env, steps):
        self.model = PPO.load(model_path, env = vec_env)
        self.env = vec_env
        self.steps = steps
        self.cca_positions = []
        self.foxtrot_positions = []

        # Initially, we do NOT grab the terrain map here,
        self.terrain_map = None
        self.grid_size = globals.grid_size
        self.block_size = 50  # This is used for terrain discretization
        self.x_blocks = None
        self.y_blocks = None

    def run_simulation(self):
        """Run the simulation and collect positions for visualization."""
        obs = self.env.reset()

        # Right after reset, the environment re-generated the terrain.
        if globals.ENABLE_TERRAIN:
            self.terrain_map = self.env.get_attr('terrain_map')[0]
            self.x_blocks = self.terrain_map.shape[0]
            self.y_blocks = self.terrain_map.shape[1]

        for _ in range(self.steps):
            # Get the action from the trained model
            action, _ = self.model.predict(obs)

            # Step the environment
            obs, reward, terminated, truncated = self.env.step(action)

            # Store positions for visualization
            self.cca_positions.append(
                np.array([np.array(pos) for pos in self.env.get_attr('cca_positions')[0]], dtype=float)
            )
            self.foxtrot_positions.append(
                np.array(self.env.get_attr('foxtrot_position')[0], dtype=float)
            )

            terminated = truncated = False #This is wrong. It's happening because terminated is being set to done success at end of each training episode

            if truncated:
                print("Episode Truncated (Non-natural endpoint (hard-stopped))")
            
            if terminated:
                print("Episode Terminated (Natural endpoint (Objective Achieved))")

    def visualize(self):
        """Visualize the simulation results."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if self.terrain_map is None:
            print("No terrain map to visualize!")
            return

        x_coords = np.arange(0, self.grid_size, self.grid_size // self.x_blocks)
        y_coords = np.arange(0, self.grid_size, self.grid_size // self.y_blocks)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        if globals.ENABLE_TERRAIN:
            self.terrain_map = self.env.get_attr('terrain_map')[0]
            z_values = self.terrain_map[:, :, 2]

        # Terrain map shape is (x_blocks, y_blocks)
        z_values = self.terrain_map[:, :, 2]  # Extract the terrain height (z_terrain) values

        # Plot the terrain surface
        ax.plot_surface(x_grid, y_grid, z_values, cmap='terrain', alpha=0.3, edgecolor='black')

        # Set up the grid
        ax.set_xlim([0, globals.grid_size])
        ax.set_ylim([0, globals.grid_size])
        ax.set_zlim([0, globals.grid_size])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.grid(False)  # Enable gridlines for better clarity

        def update(frame):
            ax.cla()

            # Replot terrain surface
            ax.plot_surface(
                x_grid, y_grid, z_values, cmap='terrain', alpha=0.5, edgecolor='black', zorder=1
            )

            ax.set_xlim([0, globals.grid_size])
            ax.set_ylim([0, globals.grid_size])
            ax.set_zlim([0, globals.grid_size])
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            # Plot CCA paths and current positions
            for i in range(NUM_CCA):
                if frame < len(self.cca_positions):
                    cca_path = np.array([pos[i] for pos in self.cca_positions[:frame]])
                    if len(cca_path) > 0:
                        ax.plot(cca_path[:, 0], cca_path[:, 1], cca_path[:, 2], color='green')
                        ax.scatter(*cca_path[-1], color='green', s=80, label=f'CCA {i + 1} Position')

            # Plot Foxtrot path and current position
            if frame < len(self.foxtrot_positions):
                foxtrot_path = np.array(self.foxtrot_positions[:frame])
                if len(foxtrot_path) > 0:
                    ax.plot(foxtrot_path[:, 0], foxtrot_path[:, 1], foxtrot_path[:, 2], color='orange')
                    ax.scatter(*foxtrot_path[-1], color='orange', s=100, label='Foxtrot Position')

            if frame == 1:  # Add the legend once
                ax.legend()

            # Adjust viewing angle for dynamic effect
            ax.view_init(elev=30, azim=frame * 0.5)
            
        anim = FuncAnimation(fig, update, frames=self.steps, interval=50)
        plt.show()


if __name__ == "__main__":

    # ----------------------
    # TOGGLE FOXTROT MODES HERE
    # ----------------------
    globals.BASIC_REWARD = False
    globals.COMPLEX_REWARD = True


    globals.STATIONARY_FOXTROT = False
    globals.RAND_POS = True
    globals.FIXED_POS = False
    globals.RECTANGULAR_FOXTROT = True
    globals.RAND_FIXED_CCA = False
    globals.PROXIMITY_CCA = True
    globals.ENABLE_TERRAIN = True

    # Initialize environment with the updated flags
    base_env = PPOEnv(grid_size=globals.grid_size, num_cca=NUM_CCA)
    dummy_env = DummyVecEnv([lambda: base_env])

    # Load the normalization stats
    vec_env = VecNormalize.load("./PPO_V2/Trained_VecNormalize.pkl", dummy_env)

    # Important: Set to evaluation mode
    vec_env.training = False
    # Typically disable reward normalization at test time
    vec_env.norm_reward = False
    # Also typically freeze observation normalization updates:
    # vec_env.eval() in SB3 2.0, or set vec_env.training=False in older versions

    # Initialize the showcase simulation
    showcase = ShowcaseSimulation(
        model_path="./PPO_V2/Trained_Model",
        vec_env=vec_env,
        steps=STEPS
    )

    # Run the simulation
    showcase.run_simulation()

    # Visualize the results
    showcase.visualize()