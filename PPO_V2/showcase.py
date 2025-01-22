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

    def run_simulation(self):
        """Run the simulation and collect positions for visualization."""
        obs = self.env.reset()

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

        # Set up the grid
        ax.set_xlim([0, globals.grid_size])
        ax.set_ylim([0, globals.grid_size])
        ax.set_zlim([0, globals.grid_size])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        def update(frame):
            ax.cla()
            ax.set_xlim([0, globals.grid_size])
            ax.set_ylim([0, globals.grid_size])
            ax.set_zlim([0, globals.grid_size])
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            for i in range(NUM_CCA):
                if frame >= len(self.cca_positions) or i >= len(self.cca_positions[0]):
                    continue

                # CCA path up to 'frame'
                cca_path = np.array([pos[i] for pos in self.cca_positions[:frame]])
                if len(cca_path) > 0:
                    ax.plot(cca_path[:, 0], cca_path[:, 1], cca_path[:, 2],
                            color='green', label=f'CCA {i + 1} Path' if frame == 1 and i == 0 else "")
                    # Plot current CCA position as a larger scatter
                    ax.scatter(*cca_path[-1], color='green', s=80)

            if frame < len(self.foxtrot_positions):
                foxtrot_path = np.array(self.foxtrot_positions[:frame])
                if len(foxtrot_path) > 0:
                    ax.plot(foxtrot_path[:, 0], foxtrot_path[:, 1], foxtrot_path[:, 2],
                            color='orange', label='Foxtrot Path' if frame == 1 else "")
                    # Plot current Foxtrot position as a larger scatter
                    ax.scatter(*foxtrot_path[-1], color='orange', s=80)

            if frame == 1:  # Just so the legend is not repeated every frame
                ax.legend()

        anim = FuncAnimation(fig, update, frames=self.steps, interval=10)
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