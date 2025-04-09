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
        if num_cca != 2: raise ValueError("This environment version requires num_cca=2")
        self.num_cca = num_cca
        self.observable_radius = 200.0 # Radius within which Foxtrot is visible

        # Simulation state
        self.current_step = 0
        self.max_steps = 1000 # Max steps per episode
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

        print(f"\nPPOEnv Initialized for Multi-Agent Partial Observability:")
        print(f"  Num CCAs: {self.num_cca}")
        print(f"  History Length: {self.observation_history_len}")
        print(f"  Observable Radius: {self.observable_radius}")
        print(f"  Action Space: {self.action_space}")
        print(f"  Observation Space: {self.observation_space}")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_potential = None
        self.was_observable_prev_step = False
        self.previous_avg_distance = None
        self.previous_cca_positions = None

        # --- Reset Foxtrot (Stationary, Random Position) ---
        # Ensure STATIONARY_FOXTROT and RAND_POS are True in globals or config
        initial_foxtrot_pos = self.np_random.integers(0, self.grid_size, size=3).astype(float)

        if self.foxtrot_obj is None:
            self.foxtrot_obj = Foxtrot("Foxtrot_0", initial_foxtrot_pos)
        else:
            self.foxtrot_obj.set_position(initial_foxtrot_pos)
            self.foxtrot_obj.path = [initial_foxtrot_pos.copy()]

        # --- Reset CCAs (Random Positions) ---
        # Ensure PROXIMITY_CCA and RAND_FIXED_CCA are False
        self.cca_objs = []
        for i in range(self.num_cca):
            initial_cca_pos = self.np_random.integers(0, self.grid_size, size=3).astype(float)
            # Ensure CCAs don't start too close to each other (simple check)
            if i > 0 and np.linalg.norm(initial_cca_pos - self.cca_objs[0].position) < self.cca_collision_radius * 2:
                 initial_cca_pos = self.np_random.integers(0, self.grid_size, size=3).astype(float) # Try again
            self.cca_objs.append(CCA(f"CCA_{i}", initial_cca_pos))
        self.previous_cca_positions = [cca.position.copy() for cca in self.cca_objs]

        # --- Reset Histories ---
        hist_len = self.observation_history_len
        self.action_history = [np.zeros((hist_len, 3), dtype=np.float32) for _ in range(self.num_cca)]
        self.cca_history = [np.tile(cca.position, (hist_len, 1)) for cca in self.cca_objs]
        # Foxtrot history starts assuming it's not observable
        initial_obs_foxtrot = PLACEHOLDER_POS
        self.foxtrot_history = np.tile(initial_obs_foxtrot, (hist_len, 1)) # Fill with placeholder initially

        if POSITIONAL_DEBUG:
            print(f"\n--- ENV RESET (Step 0) ---")
            print(f"{self.foxtrot_obj.name} at {self.foxtrot_obj.position}")
            for cca in self.cca_objs: print(f"{cca.name} at {cca.position}")
            print("--------------------------\n")

        observation = self._get_observation()
        info = {"reset_complete": True}
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
        Reward function for 2 CCAs searching (exploration) and intercepting (exploitation)
        with partial observability. Switches incentives based on visibility.
        """
        # --- Parameters (Tune these!) ---
        # Exploitation (Foxtrot Visible)
        exploit_progress_scale = 50.0   # Scale for distance reduction reward when Foxtrot is VISIBLE
        capture_bonus = 2500.0           # Bonus for successful capture

        # Exploration (Foxtrot Hidden)
        explore_separation_penalty = -0.5 # Penalty multiplier for squared distance between CCAs (encourage separation)
        explore_movement_bonus = 0.1    # Small reward multiplier for average distance moved by CCAs

        # Shared Penalties/Bonuses
        find_bonus = 500.0               # One-time bonus for finding Foxtrot
        time_penalty = -1.0              # Penalty per step
        cca_collision_penalty = -10.0    # Smaller penalty for getting too close (vs crossing radius)
        energy_penalty_coeff = 0.01

        reward = 0.0

        # --- Calculate Current State ---
        current_cca_positions = [cca.position.copy() for cca in self.cca_objs] # Copy current positions
        current_foxtrot_position = self.foxtrot_obj.position

        if not current_cca_positions or len(current_cca_positions) != self.num_cca: return 0.0 # Safety check

        # --- Shared Penalty Calculations ---
        # 1. Time Penalty
        reward += time_penalty

        # 2. Energy Penalty
        if actions is not None:
            action_magnitude_sq = np.sum(actions**2)
            reward -= energy_penalty_coeff * action_magnitude_sq

        # 3. CCA-CCA Collision Penalty
        if self.num_cca > 1:
             dist_cca = np.linalg.norm(current_cca_positions[0] - current_cca_positions[1])
             if dist_cca < self.cca_collision_radius:
                 reward += cca_collision_penalty
                 # Maybe make penalty harsher if they are *really* close
                 # reward -= 5.0 / (dist_cca + 1e-6) # Example inverse distance penalty

        # --- Visibility-Dependent Rewards ---
        current_avg_distance_to_foxtrot = np.mean([np.linalg.norm(pos - current_foxtrot_position) for pos in current_cca_positions])

        if is_observable_now:
            # --- Exploitation Phase ---
            mode = "Exploit"
            # 4a. Progress Reward (Potential Shaping for "Darting")
            progress_reward = 0.0
            if self.previous_avg_distance is not None:
                distance_reduction = self.previous_avg_distance - current_avg_distance_to_foxtrot
                progress_reward = exploit_progress_scale * distance_reduction
            reward += progress_reward

            # 5a. Find Bonus (if just became visible)
            if not self.was_observable_prev_step:
                reward += find_bonus

            # 6a. Capture Bonus Check (Termination handled in step)
            min_dist_to_foxtrot = np.linalg.norm(current_cca_positions[0] - current_foxtrot_position) if self.num_cca==1 else min(np.linalg.norm(pos - current_foxtrot_position) for pos in current_cca_positions)
            if min_dist_to_foxtrot < self.capture_radius:
                reward += capture_bonus

        else:
            # --- Exploration Phase ---
            mode = "Explore"
            # 4b. Separation Reward (Penalize proximity between CCAs)
            if self.num_cca > 1:
                 # Use squared distance? Or inverse? Let's try penalizing proximity more strongly.
                 # Inverse distance penalty: gets very large when close
                 # reward -= 1.0 / (dist_cca + 1e-6) # Very sensitive
                 # Linear penalty based on closeness:
                 proximity_penalty = max(0, (self.cca_collision_radius * 5) - dist_cca) # Penalize if within 5x collision radius
                 reward -= explore_separation_penalty * proximity_penalty # Positive penalty value

            # 5b. Movement/Coverage Bonus (Reward moving away from previous spots)
            movement_reward = 0.0
            if self.previous_cca_positions is not None:
                avg_dist_moved = np.mean([np.linalg.norm(current_cca_positions[i] - self.previous_cca_positions[i]) for i in range(self.num_cca)])
                movement_reward = explore_movement_bonus * avg_dist_moved
            reward += movement_reward

        # --- Update State for Next Step ---
        self.previous_avg_distance = current_avg_distance_to_foxtrot
        self.previous_cca_positions = current_cca_positions # Store list of current positions

        # --- Debug Printing ---
        if REWARD_DEBUG:
            dist_str = ", ".join([f"{np.linalg.norm(p - current_foxtrot_position):.1f}" for p in current_cca_positions])
            cca_dist_str = f"{dist_cca:.1f}" if self.num_cca > 1 else "N/A"
            find_b = find_bonus if is_observable_now and not self.was_observable_prev_step else 0
            capt_b = capture_bonus if is_observable_now and min_dist_to_foxtrot < self.capture_radius else 0
            cca_coll_p = cca_collision_penalty if self.num_cca > 1 and dist_cca < self.cca_collision_radius else 0
            energy_p = -energy_penalty_coeff * action_magnitude_sq if actions is not None else 0
            prog_rew = progress_reward if is_observable_now and self.previous_avg_distance is not None else 0
            sep_p = -explore_separation_penalty * proximity_penalty if not is_observable_now and self.num_cca > 1 else 0
            move_b = movement_reward if not is_observable_now and self.previous_cca_positions is not None else 0


            print(f"Step {self.current_step}: Total={reward:.2f} | Mode={mode} | Dists=[{dist_str}] | CCA_Dist={cca_dist_str} | "
                  f"Prog={prog_rew:.2f} | SepP={sep_p:.2f} | MoveB={move_b:.2f} | FindB={find_b:.0f} | CaptB={capt_b:.0f} | "
                  f"CCACollP={cca_coll_p:.0f} | EnergyP={energy_p:.2f} | TimeP={time_penalty:.0f}")

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
    env = PPOEnv(grid_size=500, num_cca=2)
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
        batch_size=600,           # Increase batch size
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

    # --- Save Model and Environment Wrapper ---
    save_dir = "./LMA_RL_MultiAgent_V1" # New save directory
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "trained_model_multi_lma")
    vec_env_path = os.path.join(save_dir, "trained_vecnormalize_multi_lma.pkl")

    # --- Workaround for TensorFlow/TensorBoard Pickle Error ---
    print("Temporarily disabling logger before saving...")
    original_logger = getattr(model, 'logger', None) # Safely get logger
    if original_logger: model.logger = None
    # --------------------------------------------------------

    try:
        print(f"Saving model to {model_path}.zip...")
        model.save(model_path)
        print(f"Saving VecNormalize state to {vec_env_path}...")
        vec_env.save(vec_env_path) # Save VecNormalize stats
        print("\nModel and VecNormalize state saved successfully!")

    except Exception as e:
        print(f"\nError during saving: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Restore logger ---
        if original_logger:
            print("Restoring logger reference...")
            model.logger = original_logger

    # --- Cleanup ---
    vec_env.close()
    print("\nTraining and saving complete.")