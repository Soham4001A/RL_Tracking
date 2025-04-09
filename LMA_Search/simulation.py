# simulation.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import pow
import math
from stable_baselines3 import PPO # Using PPO, assuming GRPO not available/needed now
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback # For potential plotting
import os
import torch
import matplotlib.pyplot as plt # For potential plotting

# Internal Module Imports
from classes import CCA, Foxtrot, LMAFeaturesExtractor # Import PPOEnv from classes now
import globals

# Define DEBUG flags and constants if not in globals
try: REWARD_DEBUG = globals.REWARD_DEBUG
except AttributeError: REWARD_DEBUG = True
try: POSITIONAL_DEBUG = globals.POSITIONAL_DEBUG
except AttributeError: POSITIONAL_DEBUG = True


# Define DEBUG flags and constants if not in globals
try: REWARD_DEBUG = globals.REWARD_DEBUG
except AttributeError: REWARD_DEBUG = True # Default to True for debugging new reward
try: POSITIONAL_DEBUG = globals.POSITIONAL_DEBUG
except AttributeError: POSITIONAL_DEBUG = True # Default True for debugging new setup

try: grid_size = globals.grid_size
except AttributeError: grid_size = 500
try: step_size = globals.step_size
except AttributeError: step_size = 10
try: spawn_range = globals.spawn_range # Used only if RAND_FIXED_CCA is True
except AttributeError: spawn_range = 50

# Placeholder for Foxtrot position when not observable
PLACEHOLDER_POS = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

class PPOEnv(gym.Env):
    """
    Custom Environment for 2 CCAs intercepting a Foxtrot with partial observability.
    Observation includes history of CCA states, actions, and Foxtrot state (conditional).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_size=500, num_cca=2): # Default num_cca=2
        super().__init__()
        self.grid_size = grid_size
        #if num_cca != 2: raise ValueError("This environment version requires num_cca=2")
        self.num_cca = num_cca
        self.observable_radius = 200.0 # Radius within which Foxtrot is visible

        # Simulation state
        self.current_step = 0
        self.max_steps = 2000 # Max steps per episode
        self.foxtrot_obj = None
        self.cca_objs = []

        # Reward Shaping State
        self.previous_potential = None
        self.previous_cca_positions = None # Reset previous positions for exploration bonus calc
        self.was_observable_prev_step = False # For detecting visibility changes
        self.previous_avg_distance = None # Store previous avg distance for shaping
        

        # --- Define Action Space (Actions for 2 CCAs) ---
        self.action_space = spaces.Box(
            low=-step_size, high=step_size, shape=(self.num_cca, 3), dtype=np.float32
        )

        # --- Define Observation Space (Longer History) ---
        self.observation_history_len = 200 # Increased history length
        # Dims per step: (CCA1_pos + CCA1_act + CCA2_pos + CCA2_act + Foxtrot_pos)
        #                 ( 3   +    3     +    3     +    3     +      3     ) = 15
        self.observation_dim_per_step = (3 + 3) * self.num_cca + 3
        self.observation_shape_total = self.observation_history_len * self.observation_dim_per_step

        # Observation bounds: Positions (0 to grid_size), Actions (-step to +step), Placeholder (-1)
        low_bound = -max(float(self.grid_size), float(step_size), 1.0) # Include -1 for placeholder
        high_bound = max(float(self.grid_size), float(step_size))

        self.observation_space = spaces.Box(
            low=low_bound, high=high_bound,
            shape=(self.observation_shape_total,), dtype=np.float32,
        )

        # --- Histories (Initialized in reset) ---
        self.action_history = None # List of numpy arrays [num_cca] x (hist_len, 3)
        self.cca_history = None    # List of numpy arrays [num_cca] x (hist_len, 3)
        self.foxtrot_history = None # Numpy array (hist_len, 3)

        # --- Reward Function Utils (Store as attributes) ---
        self.capture_radius = 15.0  # Radius for successful capture bonus/termination
        self.cca_collision_radius = 5.0 # Radius for CCA-CCA collision penalty
        self.cell_visit_counts = {} # Dictionary to store counts per grid cell
        self.grid_resolution = 20 # Size of grid cells for visit count (tune this)
        self.history_len = 200 # Store history length

        print(f"\nPPOEnv Initialized for Multi-Agent Partial Observability:")
        print(f"  Num CCAs: {self.num_cca}")
        print(f"  History Length: {self.observation_history_len}")
        print(f"  Observable Radius: {self.observable_radius}")
        print(f"  Action Space: {self.action_space}")
        print(f"  Observation Space: {self.observation_space}")

    def _pos_to_grid_cell(self, position):
        """Converts a 3D position to a discrete grid cell tuple."""
        return tuple((position // self.grid_resolution).astype(int))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_avg_distance_to_target = None # Reset for reward calc
        self.was_observable_prev_step = False

        # Reset Foxtrot (Stationary, Random Position)
        initial_foxtrot_pos = self.np_random.integers(0, self.grid_size, size=3).astype(float)
        if self.foxtrot_obj is None: self.foxtrot_obj = Foxtrot("Foxtrot_0", initial_foxtrot_pos)
        else: self.foxtrot_obj.set_position(initial_foxtrot_pos); self.foxtrot_obj.path = [initial_foxtrot_pos.copy()]

        # Reset CCAs (Random Positions)
        self.cca_objs = []
        min_start_sep = 10.0 # Ensure agents don't start exactly on top of each other
        for i in range(self.num_cca):
            while True: # Keep trying until a suitable position is found
                 initial_cca_pos = self.np_random.integers(0, self.grid_size, size=3).astype(float)
                 is_too_close = False
                 for j in range(len(self.cca_objs)):
                     if np.linalg.norm(initial_cca_pos - self.cca_objs[j].position) < min_start_sep:
                         is_too_close = True
                         break
                 if not is_too_close:
                     break # Found a good position
            self.cca_objs.append(CCA(f"CCA_{i}", initial_cca_pos))

        # Reset Histories
        self.action_history = [np.zeros((self.history_len, 3), dtype=np.float32) for _ in range(self.num_cca)]
        self.cca_history = [np.tile(cca.position, (self.history_len, 1)) for cca in self.cca_objs]
        is_observable_init = any(np.linalg.norm(cca.position - self.foxtrot_obj.position) < self.observable_radius for cca in self.cca_objs)
        initial_obs_foxtrot = self.foxtrot_obj.position if is_observable_init else PLACEHOLDER_POS
        self.foxtrot_history = np.tile(initial_obs_foxtrot, (self.history_len, 1))
        self.was_observable_prev_step = is_observable_init

        # --- Reset Exploration State ---
        self.cell_visit_counts = {}
        # Pre-populate with initial positions
        for cca in self.cca_objs:
            cell = self._pos_to_grid_cell(cca.position)
            self.cell_visit_counts[cell] = self.cell_visit_counts.get(cell, 0) + 1
        # -----------------------------

        if POSITIONAL_DEBUG: print(f"\n--- ENV RESET ---\n{self.foxtrot_obj.name} @ {self.foxtrot_obj.position}\n"+"\n".join([f"{c.name} @ {c.position}" for c in self.cca_objs])+"\n--------------")

        observation = self._get_observation()
        info = {"initial_observability": is_observable_init}
        return observation, info

    def step(self, actions):
        self.current_step += 1
        actions = np.array(actions, dtype=np.float32)
        if actions.shape != (self.num_cca, 3):
            try: actions = actions.reshape(self.num_cca, 3)
            except ValueError: raise ValueError(f"Invalid action shape: Expected ({self.num_cca}, 3), got {actions.shape}")
        clipped_actions = np.clip(actions, -step_size, step_size)

        # --- Update CCA States ---
        for i in range(self.num_cca):
            self.action_history[i] = np.roll(self.action_history[i], shift=-1, axis=0)
            self.action_history[i][-1] = clipped_actions[i]
            self.cca_objs[i].move(clipped_actions[i])
            self.cca_objs[i].position = np.clip(self.cca_objs[i].position, 0, self.grid_size - 1)
            self.cca_history[i] = np.roll(self.cca_history[i], shift=-1, axis=0)
            self.cca_history[i][-1] = self.cca_objs[i].position

        # --- Update Foxtrot State (Stationary in this version) ---
        # No movement needed for foxtrot_obj.position

        # --- Determine CURRENT observability & Update Foxtrot History ---
        is_observable_now = any(np.linalg.norm(cca.position - self.foxtrot_obj.position) < self.observable_radius for cca in self.cca_objs)
        current_foxtrot_obs = self.foxtrot_obj.position if is_observable_now else PLACEHOLDER_POS
        self.foxtrot_history = np.roll(self.foxtrot_history, shift=-1, axis=0)
        self.foxtrot_history[-1] = current_foxtrot_obs

        # --- Calculate Reward ---
        # Pass current observability status AND use stored previous status
        reward = self._calculate_reward(actions=clipped_actions, is_observable_now=is_observable_now)
        
        # --- Check Termination Conditions ---
        terminated = False # Goal reached (capture)
        truncated = False # Time limit

        min_dist = min(np.linalg.norm(cca.position - self.foxtrot_obj.position) for cca in self.cca_objs)
        if min_dist < self.capture_radius:
            #terminated = True # -> I want the episode to end naturally so we cacn get more positive reward points within the data set
            print("WITHIN CAPTURE RADIUS!")
            if REWARD_DEBUG: print(f"--- CAPTURE at step {self.current_step} ---")

        if self.current_step >= self.max_steps:
            truncated = True
            if REWARD_DEBUG and not terminated: print(f"--- Truncated at step {self.current_step} ---")

        # Update observability status for next step's reward calc
        self.was_observable_prev_step = is_observable_now

        # --- Get Observation ---
        observation = self._get_observation() # Observation uses the updated histories
        info = {"is_observable": is_observable_now, "min_dist": min_dist} # Example info

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Constructs the observation vector including potentially masked Foxtrot history."""
        state_components = []
        # History is ordered [oldest, ..., newest]
        for t in range(self.observation_history_len):
            for i in range(self.num_cca):
                state_components.append(self.cca_history[i][t])   # CCA pos @ t
                state_components.append(self.action_history[i][t]) # Action taken @ t-1 leading to state t (or 0 for t=0)
            state_components.append(self.foxtrot_history[t])   # Foxtrot pos @ t (potentially placeholder)

        obs_flat = np.concatenate(state_components).astype(np.float32)
        if obs_flat.shape != (self.observation_shape_total,):
             raise ValueError(f"Observation shape mismatch! Expected {(self.observation_shape_total,)}, got {obs_flat.shape}")
        return obs_flat

    def _calculate_reward(self, actions=None, is_observable_now=False):
        """
        Reward function with distinct normalization targets:
        - Intercept Mode (Visible): Rewards primarily in [0, 500] (Positive = good)
        - Search Mode (Hidden): Rewards primarily in [-500, 0] (Negative = bad, less negative = better)
        """
        # --- Parameters (Tune!) ---
        # Target range scales
        INTERCEPT_REWARD_SCALE = 500.0
        SEARCH_REWARD_SCALE = 500.0 # Max magnitude for negative search rewards

        # Intercept (Visible) Weights (Should sum to 1.0 for normalized range)
        w_progress = 1.0 # Weight for normalized progress towards target

        # Search (Hidden) Weights (Should sum to 1.0 for normalized range)
        w_proximity = 0.6  # Weight for proximity penalty (closer CCAs = more penalty)
        w_revisit = 0.4    # Weight for revisit penalty

        # Shared Components
        capture_bonus = 1000.0 # Keep large bonus outside normalized range
        find_bonus = 200.0     # Bonus for finding target
        time_penalty = -0.5    # Small penalty per step
        cca_collision_penalty = -10.0 # Direct collision penalty
        energy_penalty_coeff = 0.005 # Energy use penalty

        # Normalization helpers
        max_dist_change_norm = 1.0 * step_size # Approx max progress per step
        # Target separation for normalization (used to define PROXIMITY)
        # Max possible separation is grid_size*sqrt(3), avg is complex.
        # Let's normalize proximity based on collision radius instead.
        # Proximity = 1 if dist=0, 0 if dist > threshold
        proximity_threshold = self.cca_collision_radius * 10 # e.g., 50 units apart penalizes
        max_revisit_penalty_norm = 10.0 # Estimated max raw revisit penalty (tune!)

        # --- Calculate Current State ---
        current_cca_positions = [cca.position for cca in self.cca_objs]
        current_foxtrot_position = self.foxtrot_obj.position
        if not current_cca_positions or len(current_cca_positions) != self.num_cca: return 0.0

        # --- Calculate Distances ---
        current_avg_distance = np.mean([np.linalg.norm(pos - current_foxtrot_position) for pos in current_cca_positions])
        min_dist_to_foxtrot = min(np.linalg.norm(pos - current_foxtrot_position) for pos in current_cca_positions)
        avg_cca_separation = 0.0 # Not directly used in reward now, but useful for debug
        min_cca_dist = float('inf')
        if self.num_cca > 1:
            pair_count = 0
            for i in range(self.num_cca):
                for j in range(i + 1, self.num_cca):
                    dist_ij = np.linalg.norm(current_cca_positions[i] - current_cca_positions[j])
                    avg_cca_separation += dist_ij
                    min_cca_dist = min(min_cca_dist, dist_ij)
                    pair_count += 1
            if pair_count > 0: avg_cca_separation /= pair_count

        # --- Initialize Reward ---
        reward = 0.0
        mode_reward_scaled = 0.0 # Store the scaled mode-specific reward for debugging

        # --- Mode-Dependent Rewards ---
        if is_observable_now:
            # --- Mode 1: Intercept ---
            mode = "Intercept"
            # Calculate normalized progress reward [0, w_progress]
            progress_reward_normalized_component = 0.0
            if self.previous_avg_distance_to_target is not None:
                distance_reduction = self.previous_avg_distance_to_target - current_avg_distance
                normalized_progress = np.clip(distance_reduction / (max_dist_change_norm + 1e-6), -1.0, 1.0)
                progress_reward_normalized_component = w_progress * (normalized_progress + 1.0) / 2.0

            # Scale to target range [0, 500 * w_progress]
            mode_reward_scaled = progress_reward_normalized_component * INTERCEPT_REWARD_SCALE
            reward += mode_reward_scaled

            # Add find bonus (one-time)
            if not self.was_observable_prev_step: reward += find_bonus

        else:
            # --- Mode 2: Search ---
            mode = "Search"
            # 1. Proximity Penalty Score (Normalized [0, 1], higher means closer/worse)
            proximity_score_normalized = 0.0
            if self.num_cca > 1:
                 # Linearly scale penalty from 1 (at collision radius) down to 0 (at threshold)
                 proximity_score_normalized = np.clip(1.0 - (min_cca_dist - self.cca_collision_radius) / (proximity_threshold - self.cca_collision_radius + 1e-6), 0.0, 1.0)

            # 2. Re-exploration Penalty Score (Normalized [0, 1], higher means more revisited)
            revisit_penalty_raw = 0.0
            current_cells = set()
            for pos in current_cca_positions:
                cell = self._pos_to_grid_cell(pos)
                if cell not in current_cells:
                    visit_count = self.cell_visit_counts.get(cell, 0)
                    revisit_penalty_raw += math.sqrt(visit_count) # Penalty grows with visits
                    self.cell_visit_counts[cell] = visit_count + 1
                    current_cells.add(cell)
            # Normalize the raw penalty based on expected max
            revisit_score_normalized = np.clip(revisit_penalty_raw / (max_revisit_penalty_norm * self.num_cca + 1e-6), 0.0, 1.0)

            # Calculate total *negative* reward for this mode, scaled to [-500 * (w_prox + w_revisit), 0]
            proximity_penalty_scaled = - (w_proximity * proximity_score_normalized * SEARCH_REWARD_SCALE)
            revisit_penalty_scaled = - (w_revisit * revisit_score_normalized * SEARCH_REWARD_SCALE)
            mode_reward_scaled = proximity_penalty_scaled + revisit_penalty_scaled
            reward += mode_reward_scaled


        # --- Shared Components (Applied AFTER mode rewards) ---

        # Capture Bonus (Can happen if visible)
        if min_dist_to_foxtrot < self.capture_radius:
            reward += capture_bonus # Added on top of any mode reward

        # Add essential penalties AFTER scaling mode rewards
        reward += time_penalty
        if actions is not None:
            action_magnitude_sq = np.sum(actions**2)
            reward -= energy_penalty_coeff * action_magnitude_sq
        if self.num_cca > 1 and min_cca_dist < self.cca_collision_radius:
            reward += cca_collision_penalty # Direct collision is heavily penalized

        # --- Update State for Next Step ---
        self.previous_avg_distance_to_target = current_avg_distance
        # was_observable_prev_step is updated in step() method

        # --- Debug Printing ---
        if REWARD_DEBUG:
             dist_str = f"AvgD:{current_avg_distance:.1f},MinD:{min_dist_to_foxtrot:.1f}"
             cca_dist_str = f"AvgSep:{avg_cca_separation:.1f},MinSep:{min_cca_dist:.1f}" if self.num_cca > 1 else "N/A"
             obs_status = "Visible" if is_observable_now else "Hidden "
             capt_b = capture_bonus if min_dist_to_foxtrot < self.capture_radius else 0
             find_b = find_bonus if is_observable_now and not self.was_observable_prev_step else 0
             cca_coll_p_direct = cca_collision_penalty if self.num_cca > 1 and min_cca_dist < self.cca_collision_radius else 0
             energy_p = -energy_penalty_coeff * action_magnitude_sq if actions is not None else 0

             prog_r_pr=0.0; sep_p_pr=0.0; rev_p_pr=0.0 # For printing only
             if is_observable_now:
                 prog_r_pr = mode_reward_scaled # Intercept mode reward
             else:
                 sep_p_pr = proximity_penalty_scaled # Search mode penalty part 1
                 rev_p_pr = revisit_penalty_scaled   # Search mode penalty part 2

             print(f"Step {self.current_step}: Total={reward:.2f} | M={mode} | Dists=[{dist_str}] | CCAs=[{cca_dist_str}] | "
                   f"ModeRew={mode_reward_scaled:.1f} (ProgR={prog_r_pr:.1f} | ProxP={sep_p_pr:.1f} | RevisP={rev_p_pr:.1f}) | "
                   f"FindB={find_b:.0f} | CaptB={capt_b:.0f} | CCACollP={cca_coll_p_direct:.0f} | EnergyP={energy_p:.2f} | TimeP={time_penalty:.1f}")


        return float(reward)
    
#===============================================
# Optional Plotting Callback (Keep if desired)
#===============================================
# (PlottingCallback class definition can be placed here or imported)
# Example placeholder:
class PlottingCallback(BaseCallback):
     def __init__(self, plot_freq: int, log_dir: str, verbose: int = 1):
         super().__init__(verbose)
         self.plot_freq = plot_freq; self.log_dir = log_dir; self.save_path = os.path.join(log_dir, "monitor.csv")
         self.fig = None; self.ax = None; self.steps = []; self.rewards = []
     def _on_training_start(self) -> None: pass # Implement full plotting if needed§
     def _on_step(self) -> bool: return True
     def _on_training_end(self) -> None: pass

#===============================================
# Main Training Block
#===============================================
if __name__ == "__main__":

    def linear_schedule(initial_value: float):
        """Linear schedule function for learning rate."""
        def schedule(progress_remaining: float):
            return progress_remaining * initial_value
        return schedule

    # --- Set Training Configuration Flags ---
    # Environment Setup
    globals.STATIONARY_FOXTROT = True  # Target is stationary
    globals.RECTANGULAR_FOXTROT = False # Target is not moving in a rectangle
    globals.RAND_POS = True           # Target spawns randomly
    globals.FIXED_POS = False         # Target does not spawn fixed
    globals.PROXIMITY_CCA = False     # CCAs start randomly
    globals.RAND_FIXED_CCA = False    # CCAs dont start near target

    # Reward Setup
    globals.COMPLEX_REWARD = True     # Use the new complex reward
    globals.BASIC_REWARD = False

    # --- Create and wrap the environment ---
    #log_dir = "./sb3_logs_multiagent/" # Directory for Monitor logs
    #os.makedirs(log_dir, exist_ok=True)

    # Instantiate environment with 2 CCAs
    env = PPOEnv(grid_size=500, num_cca=4)
    #env = Monitor(env, log_dir) # Wrap with Monitor BEFORE DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    # IMPORTANT: Normalize observations. May need adjustment for placeholder value (-1).
    # Consider normalizing reward as well.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=100.0) # Increased clip_obs range


    # --- Define LMA Hyperparameters ---
    # ** TUNE THESE CAREFULLY for L=200 **
    lma_kwargs = dict(
        embed_dim=128,          # d0: Increased initial embedding?
        num_heads_stacking=8,   # nh: (128 % 8 == 0) -> dk=16
        target_l_new=50,        # Target L_new (L=200) -> find divisor for 200*128=25600
                                # 50 is divisor. L_new=50 -> C_new=512
        d_new=64,               # d_new: Latent dimension
        num_heads_latent=8,     # Latent heads (64 % 8 == 0) -> latent_dk=8
        ff_latent_hidden=128*3,   # Latent MLP hidden (e.g., 2*d_new)
        num_lma_layers=3,       # More layers for longer sequence and harder task
        seq_len=100, # Should be 200
        dropout=0.1,
        bias=True
    )

    # --- Define Policy Keyword Arguments ---
    # Input to MLP head is L_new * d_new = 50 * 64 = 3200
    policy_kwargs = dict(
        features_extractor_class=LMAFeaturesExtractor,
        features_extractor_kwargs=lma_kwargs,
        # Adjust net_arch based on new feature dim (3200) - maybe needs more capacity
        net_arch=dict(pi=[256, 128], vf=[256, 128]) # Example larger arch
    )

    # --- Define PPO Model ---
    model_class = PPO
    model_specific_kwargs = {}

    # --- Directory for Tensorboard Logs ---
    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, "..")
    absolute_parent_dir = os.path.abspath(parent_dir)  

    model = model_class(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        normalize_advantage=True,
        learning_rate=linear_schedule(initial_value= 0.0003), # Possibly lower LR for harder task
        n_steps=2000,             # Increase steps for more data per update
        batch_size=1000,           # Increase batch size
        n_epochs=3,
        gamma=0.85,               # Standard discount factor
        gae_lambda=0.8,           # Standard GAE factor
        clip_range=0.2,           # Standard PPO clip range
        ent_coef=0.001,           # Small entropy encourages exploration slightly
        vf_coef=0.8,              # Value function coefficient
        max_grad_norm=0.5,
        tensorboard_log= absolute_parent_dir + "/TensorBoardLogs/",
        **model_specific_kwargs
    )

    # --- Optional: Display Model Summary ---
    try:
        from torchinfo import summary
        # Need to create a dummy observation matching the space
        # Use vec_env.observation_space which reflects normalization wrapper if applied
        summary(model.policy, input_size=(vec_env.num_envs, *vec_env.observation_space.shape))
    except ImportError:
        print("torchinfo not found, skipping model summary.")
    except Exception as e:
        print(f"Error getting model summary: {e}")

    proceed = input("Review model summary and config. Continue with training? (y/n): ")
    if proceed.lower() != 'y':
        print("Training aborted.")
        vec_env.close() # Close env if aborting
        exit()

    # --- Training ---
    print("\nStarting Training (Multi-Agent, Partial Observability)...")
    # No curriculum for now, just train on the main task
    total_train_steps = 200_000 # Define total steps for this phase

    # Instantiate callbacks (optional plotting or others)
    # plot_callback = PlottingCallback(plot_freq=2048, log_dir=log_dir) # Example
    callback_list = None # Or [plot_callback]

    try:
        model.learn(
            total_timesteps=total_train_steps,
            callback=callback_list,
            reset_num_timesteps=True # Start counting timesteps from 0
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    # Add other exception handling if needed

    # --- Save Model and Environment Wrapper ---)

    # --- Workaround for TensorFlow/TensorBoard Pickle Error ---
    #print("Temporarily disabling logger before saving...")
    #original_logger = getattr(model, 'logger', None) # Safely get logger
    #if original_logger: model.logger = None
    # --------------------------------------------------------

    try:
        # --- Save Model and Environment Wrapper ---
        model.save("Trained_Model")
        vec_env.save("Trained_VecNormalize.pkl")
        print("Model Saved Succesfully!")

    except Exception as e:
        print(f"\nError during saving: {e}")
        import traceback
        traceback.print_exc()
    #finally:
        # --- Restore logger ---
    #    if original_logger:
    #        print("Restoring logger reference...")
    #        model.logger = original_logger

    # --- Cleanup ---
    vec_env.close()
    print("\nTraining and saving complete.")