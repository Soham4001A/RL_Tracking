import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter  # Added PillowWriter for GIF saving
import os

# Stable Baselines and Environment Import
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from simulation import PPOSwarmCurriculumEnv  # Import the PPOEnv class from the simulation module
import globals  # Import globals to set flags

# --- Visualization Parameters ---
STEPS = 10000  # Number of steps to visualize (adjust as needed)
STEP_SIZE = globals.step_size
GRID_SIZE = globals.grid_size  # Get grid size from globals
NUM_CCA = 4
HISTORY_LEN = 10
FORMATION_SCALE = 40.0

# --- Model and VecNormalize Paths ---
# **** UPDATE THESE PATHS to where your trained model and VecNormalize stats are saved ****
MODEL_PATH = "trained_swarm_curriculum_LMA/swarm_curriculum_ppo_model"
VEC_NORM_PATH = "trained_swarm_curriculum_LMA/swarm_curriculum_vecnormalize.pkl"

class ShowcaseSimulation:
    def __init__(self, model_path, vec_normalize_path, steps):
        self.model_path = model_path
        self.vec_normalize_path = vec_normalize_path
        self.steps = steps
        self.cca_positions_history = []  # List to store positions of ALL CCAs at each step
        self.foxtrot_positions_history = []  # List to store Foxtrot position at each step
        self.env = self._load_env_and_model()  # Load env and model during init

    def _load_env_and_model(self):
        """Loads the VecNormalize statistics and the PPO model."""
        # Create the base environment with the same parameters used for training
        print("Creating base environment for loading...")
        base_env = PPOSwarmCurriculumEnv(grid_size=GRID_SIZE, num_cca=NUM_CCA, history_len=HISTORY_LEN,
                                step_size_env=STEP_SIZE, formation_scale=FORMATION_SCALE)
        dummy_env = DummyVecEnv([lambda: base_env])

        # Load the normalization stats using the correct os.path.exists call
        print(f"Loading VecNormalize stats from: {self.vec_normalize_path}")
        if not os.path.exists(self.vec_normalize_path):
            raise FileNotFoundError(f"VecNormalize file not found at {self.vec_normalize_path}")
        vec_env = VecNormalize.load(self.vec_normalize_path, dummy_env)

        # Set to evaluation mode (IMPORTANT!)
        vec_env.training = False
        # Don't update normalization stats during evaluation; keep normalizing observations based on loaded stats
        vec_env.norm_obs = True
        vec_env.norm_reward = False  # Usually don't normalize reward for evaluation interpretation

        print(f"Loading trained model from: {self.model_path}.zip")
        if not os.path.exists(f"{self.model_path}.zip"):
            raise FileNotFoundError(f"Model file not found at {self.model_path}.zip")
        # Pass the loaded vec_env to the model loader
        self.model = PPO.load(self.model_path, env=vec_env)
        print("Model and VecNormalize loaded successfully.")
        return vec_env  # Return the loaded and configured vec_env

    def run_simulation(self):
        """Run the simulation using the loaded model and collect data."""
        print(f"Running simulation for {self.steps} steps...")
        self.cca_positions_history = []
        self.foxtrot_positions_history = []

        # Unpack the tuple returned by reset to get the observation (ignoring info)
        obs = self.env.reset()

        for step_num in range(self.steps):
            action, _ = self.model.predict(obs, deterministic=True)

            # Step through the environment; unpack the returned tuple
            obs, reward, dones, infos = self.env.step(action)

            # Use env.get_attr to fetch the current CCA and Foxtrot objects
            current_cca_objs = self.env.get_attr('cca_objs')[0]
            current_foxtrot_obj = self.env.get_attr('foxtrot_obj')[0]

            self.cca_positions_history.append(
                np.array([cca.position.copy() for cca in current_cca_objs], dtype=float)
            )
            self.foxtrot_positions_history.append(
                current_foxtrot_obj.position.copy()
            )

            # Check if any episode is done
            is_done = any(dones)
            if step_num % 100 == 0 or is_done:
                info_dict = infos[0] if infos else {}
                term_reason = "Unknown"
                if info_dict.get("TimeLimit.truncated", False):
                    term_reason = "Truncated (TimeLimit)"
                elif info_dict.get("terminated", False):
                    term_reason = "Terminated (Goal?)"
                elif is_done:
                    term_reason = "Done (Unknown)"
                print(f"Step: {step_num}, Reward: {reward[0]:.2f}, Done: {is_done} ({term_reason})")

            if is_done:
                print(f"Episode finished at step {step_num + 1}.")
                break  # Stop simulation if the episode is finished
        print("Simulation run complete.")

    def visualize(self):
        """Visualize the simulation results using matplotlib animation."""
        if not self.cca_positions_history or not self.foxtrot_positions_history:
            print("No simulation data to visualize. Run run_simulation() first.")
            return

        print("Preparing visualization...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Set plot bounds based on grid size
        max_range = GRID_SIZE
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.set_zlim([0, max_range])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        cca_colors = plt.cm.viridis(np.linspace(0, 0.8, NUM_CCA))
        # Initialize plot elements for CCAs and Foxtrot
        cca_lines = [ax.plot([], [], [], lw=1, color=cca_colors[i], alpha=0.6)[0] for i in range(NUM_CCA)]
        cca_points = [ax.plot([], [], [], marker='o', markersize=6, color=cca_colors[i], linestyle='None')[0] for i in range(NUM_CCA)]
        foxtrot_line = ax.plot([], [], [], lw=1.5, color='orange', alpha=0.8)[0]
        foxtrot_point = ax.plot([], [], [], marker='*', markersize=10, color='red', linestyle='None')[0]

        # Store legend handles
        legend_handles = [plt.Line2D([0], [0], color=cca_colors[i], lw=2, label=f'CCA {i+1}') for i in range(NUM_CCA)]
        legend_handles.append(plt.Line2D([0], [0], color='orange', lw=2, label='Foxtrot'))

        def init_plot():
            """Initialize plot elements for animation."""
            for i in range(NUM_CCA):
                cca_lines[i].set_data([], [])
                cca_lines[i].set_3d_properties([])
                cca_points[i].set_data([], [])
                cca_points[i].set_3d_properties([])
            foxtrot_line.set_data([], [])
            foxtrot_line.set_3d_properties([])
            foxtrot_point.set_data([], [])
            foxtrot_point.set_3d_properties([])
            ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1))
            return cca_lines + cca_points + [foxtrot_line, foxtrot_point]

        def update(frame):
            """Update plot elements for each frame."""
            # Update Foxtrot path
            foxtrot_path_data = np.array(self.foxtrot_positions_history[:frame+1])
            if foxtrot_path_data.size > 0:
                foxtrot_line.set_data(foxtrot_path_data[:, 0], foxtrot_path_data[:, 1])
                foxtrot_line.set_3d_properties(foxtrot_path_data[:, 2])
                foxtrot_point.set_data([foxtrot_path_data[-1, 0]], [foxtrot_path_data[-1, 1]])
                foxtrot_point.set_3d_properties([foxtrot_path_data[-1, 2]])

            # Update CCAs
            for i in range(NUM_CCA):
                cca_path_data = np.array([step_positions[i] for step_positions in self.cca_positions_history[:frame+1]])
                if cca_path_data.size > 0:
                    cca_lines[i].set_data(cca_path_data[:, 0], cca_path_data[:, 1])
                    cca_lines[i].set_3d_properties(cca_path_data[:, 2])
                    cca_points[i].set_data([cca_path_data[-1, 0]], [cca_path_data[-1, 1]])
                    cca_points[i].set_3d_properties([cca_path_data[-1, 2]])

            ax.set_title(f'Simulation Step: {frame + 1}/{len(self.cca_positions_history)}')
            ax.view_init(elev=30, azim=frame * 0.3)  # Adjust rotation speed if needed
            return cca_lines + cca_points + [foxtrot_line, foxtrot_point]

        num_frames = len(self.cca_positions_history)
        print(f"Creating animation with {num_frames} frames...")
        anim = FuncAnimation(fig, update, frames=num_frames, init_func=init_plot,
                             interval=50, blit=False, repeat=False)

        print("Displaying simulation animation...")
        plt.tight_layout()
        plt.show()
        print("Animation window closed.")

if __name__ == "__main__":
    # ----------------------
    # Set globals for evaluation - these should match the final training configuration.
    # ----------------------
    globals.BASIC_REWARD = False
    globals.COMPLEX_REWARD = True

    globals.STATIONARY_FOXTROT = False  # Visualize stationary target
    globals.RECTANGULAR_FOXTROT = True
    globals.RAND_POS = True
    globals.FIXED_POS = False

    globals.PROXIMITY_CCA = False  # Use random start positions for CCAs
    globals.RAND_FIXED_CCA = False

    try:
        showcase = ShowcaseSimulation(
            model_path=MODEL_PATH,
            vec_normalize_path=VEC_NORM_PATH,
            steps=STEPS
        )
        showcase.run_simulation()
        showcase.visualize()
    except FileNotFoundError as e:
        print(f"\nError loading files: {e}")
        print("Please ensure the model and VecNormalize files exist at the specified paths:")
        print(f"Model: {MODEL_PATH}.zip")
        print(f"VecNormalize: {VEC_NORM_PATH}")
    except Exception as e:
        print(f"\nAn error occurred during showcase: {e}")
        import traceback
        traceback.print_exc()