
# THIS SCENARIO IS CURRENTLY A WORK IN PROGRESS

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import pow
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import torch

# Internal Module Imports
from classes import CCA, Foxtrot, LMAFeaturesExtractor # Import LMA extractor
import globals

# Define DEBUG flags and constants if not in globals
try: REWARD_DEBUG = globals.REWARD_DEBUG
except AttributeError: REWARD_DEBUG = True
try: POSITIONAL_DEBUG = globals.POSITIONAL_DEBUG
except AttributeError: POSITIONAL_DEBUG = True

try: grid_size = globals.grid_size
except AttributeError: grid_size = 500
try: step_size = globals.step_size
except AttributeError: step_size = 10
try: spawn_range = globals.spawn_range
except AttributeError: spawn_range = 50

# Placeholder for hidden Foxtrot position in observation
PLACEHOLDER_POS = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

#===============================================
# PPO Environment Definition
#===============================================
class PPOEnv(gym.Env):
    """
    Env for 4 CCAs, partial observability, L=200 history.
    Simplified reward: Distance penalty + Separation bonus.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_size=500, num_cca=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_cca = num_cca
        self.observable_radius = 200.0
        self.history_len = 30
        self.grid_resolution = 25  # For exploration grid cells
        self.current_step = 0
        self.max_steps = 800
        self.foxtrot_obj = None
        self.cca_objs = []
        self.visited_cells = set()  # Track visited cells
        self.action_space = spaces.Box(low=-step_size, high=step_size, shape=(self.num_cca, 3), dtype=np.float32)
        self.capture_radius = 15.0
        self.cca_collision_radius = 5.0
        self.previous_cca_positions = None
        self.action_history = None
        self.cca_history = None
        self.foxtrot_history = None

        # Observation space: Includes 4 dims for Foxtrot (pos + vis flag) per step
        self.observation_dim_per_step = (3 + 3) * self.num_cca + 4 # (pos+act)*4 + (fox_pos+vis) = 28
        self.observation_shape_total = self.history_len * self.observation_dim_per_step # 200 * 28 = 5600

        # Define observation bounds carefully
        low_bounds = np.full((self.observation_shape_total,), -max(float(self.grid_size), float(step_size), 1.0), dtype=np.float32)
        high_bounds = np.full((self.observation_shape_total,), max(float(self.grid_size), float(step_size)), dtype=np.float32)
        # Set bounds for visibility flag (0 or 1) within each step's data
        for t in range(self.history_len):
            # Calculate start index of Foxtrot data for timestep t
            step_start_index = t * self.observation_dim_per_step
            fox_data_start_index = step_start_index + (3 + 3) * self.num_cca
            vis_flag_index = fox_data_start_index + 3 # Index of the 4th element (vis flag)
            if vis_flag_index < self.observation_shape_total: # Bounds check
                 low_bounds[vis_flag_index] = 0.0
                 high_bounds[vis_flag_index] = 1.0
            else:
                 print(f"Warning: Visibility flag index calculation error? Index {vis_flag_index} >= total shape {self.observation_shape_total}")


        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds,
            shape=(self.observation_shape_total,), dtype=np.float32,
        )

        # Reward/State Tracking
        self.capture_radius = 15.0
        self.cca_collision_radius = 5.0
        self.previous_cca_positions = None

        # Histories
        self.action_history = None; self.cca_history = None; self.foxtrot_history = None

        print(f"\nPPOEnv Initialized (4 Agents, Partial Obs+VisFlag, L={self.history_len}):")
        print(f"  Obs Dim Per Step: {self.observation_dim_per_step}")
        print(f"  Total Obs Dim: {self.observation_shape_total}")
    
    def _pos_to_grid_cell(self, position):
        """Convert position to grid cell for exploration tracking."""
        return tuple((position // self.grid_resolution).astype(int))
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.visited_cells.clear()  # Reset visited cells per episode
        self.previous_cca_positions = None

        # Reset Foxtrot
        initial_foxtrot_pos = self.np_random.integers(0, self.grid_size, size=3).astype(float)
        if self.foxtrot_obj is None: self.foxtrot_obj=Foxtrot("Foxtrot_0", initial_foxtrot_pos)
        else: self.foxtrot_obj.set_position(initial_foxtrot_pos); self.foxtrot_obj.path=[initial_foxtrot_pos.copy()]

        # Reset CCAs
        self.cca_objs = []; min_start_sep = 10.0
        for i in range(self.num_cca):
            attempts=0
            while attempts < 100:
                 initial_cca_pos = self.np_random.integers(0, self.grid_size, size=3).astype(float)
                 if not any(np.linalg.norm(initial_cca_pos - cca.position) < min_start_sep for cca in self.cca_objs): break
                 attempts+=1
            if attempts==100: print("Warning: Could not place all CCAs without overlap!")
            self.cca_objs.append(CCA(f"CCA_{i}", initial_cca_pos))

        # Reset Histories
        self.action_history=[np.zeros((self.history_len,3),dtype=np.float32) for _ in range(self.num_cca)]
        self.cca_history=[np.tile(cca.position,(self.history_len,1)) for cca in self.cca_objs]

        # --- CORRECT Foxtrot History Initialization (Shape: hist_len, 4) ---
        is_observable_init = any(np.linalg.norm(cca.position - self.foxtrot_obj.position) < self.observable_radius for cca in self.cca_objs)
        initial_fox_pos_obs = self.foxtrot_obj.position if is_observable_init else PLACEHOLDER_POS # Shape (3,)
        initial_fox_vis_flag = np.array([1.0 if is_observable_init else 0.0], dtype=np.float32) # Shape (1,)
        initial_fox_obs_step = np.concatenate((initial_fox_pos_obs, initial_fox_vis_flag)) # Shape (4,)
        self.foxtrot_history = np.tile(initial_fox_obs_step, (self.history_len, 1)) # Shape (hist_len, 4)
        # --------------------------------------------------------------------

        self.previous_cca_positions = [cca.position.copy() for cca in self.cca_objs]

        if POSITIONAL_DEBUG: print(f"\n--- ENV RESET ---\n{self.foxtrot_obj.name} @ {self.foxtrot_obj.position}\n"+"\n".join([f"{c.name} @ {c.position}" for c in self.cca_objs])+"\n--------------")

        observation=self._get_observation(); info={"initial_observability": is_observable_init}
        return observation, info

    def step(self, actions):
        self.current_step += 1
        actions = np.array(actions, dtype=np.float32)
        if actions.shape != (self.num_cca, 3):
            try: actions = actions.reshape(self.num_cca, 3)
            except ValueError: raise ValueError(f"Invalid action shape: Expected ({self.num_cca},3), got {actions.shape}")
        clipped_actions = np.clip(actions, -step_size, step_size)

        current_cca_positions = []
        for i in range(self.num_cca):
            self.action_history[i]=np.roll(self.action_history[i],shift=-1,axis=0); self.action_history[i][-1]=clipped_actions[i]
            self.cca_objs[i].move(clipped_actions[i]); self.cca_objs[i].position=np.clip(self.cca_objs[i].position,0,self.grid_size-1)
            self.cca_history[i]=np.roll(self.cca_history[i],shift=-1,axis=0); self.cca_history[i][-1]=self.cca_objs[i].position
            current_cca_positions.append(self.cca_objs[i].position.copy())

        # Update Foxtrot (Stationary)

        # --- CORRECT Foxtrot History Update ---
        is_observable_now = any(np.linalg.norm(cca_pos - self.foxtrot_obj.position) < self.observable_radius for cca_pos in current_cca_positions)
        current_foxtrot_pos_obs = self.foxtrot_obj.position if is_observable_now else PLACEHOLDER_POS # Shape (3,)
        current_foxtrot_vis_flag = np.array([1.0 if is_observable_now else 0.0], dtype=np.float32) # Shape (1,)
        # Combine position and flag into the 4D observation for this step
        current_fox_obs_step = np.concatenate((current_foxtrot_pos_obs, current_foxtrot_vis_flag)) # Shape (4,)

        self.foxtrot_history = np.roll(self.foxtrot_history, shift=-1, axis=0) # Roll history
        self.foxtrot_history[-1] = current_fox_obs_step # Assign the 4D vector
        # ----------------------------------------

        # Calculate Reward
        reward = self._calculate_reward(actions=clipped_actions, current_cca_positions=current_cca_positions)

        # Check Termination/Truncation
        terminated=False; truncated=False
        min_dist_now = min(np.linalg.norm(pos - self.foxtrot_obj.position) for pos in current_cca_positions)
        if min_dist_now < self.capture_radius:
            terminated = True
            if REWARD_DEBUG: print(f"--- CAPTURE at step {self.current_step} ---")
        if self.current_step >= self.max_steps:
            truncated = True
            if REWARD_DEBUG and not terminated: print(f"--- Truncated at step {self.current_step} ---")

        # Update previous positions state AFTER using it in reward calculation
        self.previous_cca_positions = current_cca_positions

        observation = self._get_observation()
        info = {"is_observable": is_observable_now, "min_dist": min_dist_now}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        state_components = []
        for t in range(self.history_len):
            for i in range(self.num_cca):
                state_components.append(self.cca_history[i][t])   # CCA pos (3)
                state_components.append(self.action_history[i][t]) # CCA action (3)
            state_components.append(self.foxtrot_history[t])   # Foxtrot obs (pos+vis = 4)
        obs_flat = np.concatenate(state_components).astype(np.float32)
        if obs_flat.shape != (self.observation_shape_total,): raise ValueError(f"Obs shape mismatch! Exp {(self.observation_shape_total,)}, Got {obs_flat.shape}")
        return obs_flat

    def _calculate_reward(self, actions, current_cca_positions):
        """ 
        Adjusted Reward Function focusing solely on INTERCEPT mode with reduced baseline.
        """
        # Reduced baseline reward to allow intercept-related penalties and rewards to dominate.
        BASE_REWARD = 0.0
        reward = BASE_REWARD

        # Compute distances from each CCA to the foxtrot
        true_foxtrot = self.foxtrot_obj.position
        if not current_cca_positions:
            return reward

        distances = np.array([np.linalg.norm(pos - true_foxtrot) for pos in current_cca_positions])
        min_dist = distances.min()
        max_dist = distances.max()
        avg_dist = distances.mean()

        # Compute pairwise separations for formation assessment
        agent_seps = []
        if self.num_cca > 1:
            for i in range(self.num_cca):
                for j in range(i + 1, self.num_cca):
                    agent_seps.append(np.linalg.norm(current_cca_positions[i] - current_cca_positions[j]))
            avg_sep = np.mean(agent_seps) if agent_seps else float('inf')
        else:
            avg_sep = float('inf')

        # Use the intercept reward calculation exclusively
        #reward += self._calculate_intercept_reward(actions, distances, min_dist, max_dist, avg_dist, avg_sep)
        reward += self._intercept_simple_reward(actions, current_cca_positions)

        if REWARD_DEBUG:
            debug_str = (f"Focus: INTERCEPT Only | Total Reward: {reward:.2f} | Distances - min: {min_dist:.1f}, "
                         f"avg: {avg_dist:.1f}, max: {max_dist:.1f} | Avg Separation: {avg_sep:.1f}")
            print(f"Step {self.current_step}: {debug_str}")

        return float(reward)
    
    def _calculate_exploration_reward(self, actions, current_cca_positions):
        """Reward for searching when Foxtrot is not observable."""
        reward = 0.0

        # New cell bonus
        for pos in current_cca_positions:
            cell = self._pos_to_grid_cell(pos)
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                reward += 10.0

        # Separation bonus
        if self.num_cca > 1:
            separations = [np.linalg.norm(current_cca_positions[i] - current_cca_positions[j])
                           for i in range(self.num_cca) for j in range(i + 1, self.num_cca)]
            avg_sep = np.mean(separations) if separations else 0
            reward += 0.1 * avg_sep

        # Energy penalty
        energy_use = np.sum(actions ** 2)
        reward -= 0.01 * energy_use

        return reward

    def _intercept_simple_reward(self, actions,current_cca_positions):
        """
        Simplified reward function matching reference with minimal formation adaptation.
        """
        # Initialize reward
        reward = 0.0
        
        # Hyperparameters (identical to reference)
        alpha = 300           # Weight for progress
        beta = 0.05              # Weight for energy efficiency
        gamma_collision = -2000.0  # Penalty for collisions
        gamma = 0.1              # Potential shaping weight
        capture_radius = 15.0    # Radius for capture bonus
        
        target_position = self.foxtrot_obj.position
                 
        # Progress-Based Reward (simplified to match reference)
        total_progress = 0.0
        for i in range(self.num_cca):
            # Calculate progress toward target position (simpler)
            current_distance = np.linalg.norm((current_cca_positions[i]) - (target_position))
            prev_distance = np.linalg.norm((self.previous_cca_positions[i]) - 
                                        (target_position))
            
            # If the distance increased, we penalize
            if prev_distance <= current_distance:
                negative_progress = abs(current_distance - prev_distance)
                reward -= negative_progress * alpha
                total_progress -= negative_progress
            # If the distance decreased, we reward
            elif prev_distance > current_distance:
                positive_progress = abs(prev_distance - current_distance)
                reward += positive_progress * alpha
                total_progress += positive_progress
            
            # Capture bonus (identical to reference)
            if current_distance < capture_radius:
                reward += 1000.0
        
        # Energy Efficiency Penalty (identical to reference)
        energy_penalty = beta * np.sum(np.linalg.norm(actions, axis=1))
        reward -= energy_penalty
        
        # Collision Penalty (simplified) #THIS IS BROKEN
        # collision_penalty = 0.0
        # if self.num_cca > 1:
        #     for i in range(self.num_cca):
        #         for j in range(i + 1, self.num_cca):
        #             pos_i = np.asarray(current_cca_positions[i])
        #             pos_j = np.asarray(current_cca_positions[j])
        #             sep = np.linalg.norm(pos_i - pos_j)
        #             if sep < self.cca_collision_radius:
        #                 collision_penalty += gamma_collision
        
        # reward += collision_penalty
        
        # Clip reward to prevent extreme values (identical to reference)
        reward = np.clip(reward, float(-2000*self.num_cca), float(2000.0*self.num_cca))
        
        # Debug output
        if REWARD_DEBUG and self.current_step % 1 == 0:
            distances_str = ", ".join([f"{np.linalg.norm(np.asarray(current_cca_positions[i]) - np.asarray(target_position)):.2f}" for i in range(self.num_cca)])
            #print(f"Distances to targets: [{distances_str}], Raw Reward: {reward}, Progress Reward: {total_progress}")
        
        return float(reward)
    
    def _calculate_intercept_reward(self, actions, distances, min_dist, max_dist, avg_dist, avg_sep):
        """
        Calculate additional rewards for INTERCEPT mode.
        """
        # Coefficients for intercept mode
        avg_progress_coeff = 500.0
        max_progress_coeff = 250.0
        variance_penalty_coeff = 150.0
        min_dist_penalty_coeff = 0.5
        capture_bonus = 1000.0
        partial_capture_coeff = 300.0
        all_agents_capture_bonus = 500.0
        surround_coeff = 150.0
        laggard_coeff = 200.0
        time_penalty = -2.0
        energy_penalty_coeff = 2.0
        collision_penalty = -100.0

        reward = 0.0

        # Progress rewards based on distance reduction
        if self.previous_cca_positions is not None:
            prev_distances = np.array([np.linalg.norm(pos - self.foxtrot_obj.position) for pos in self.previous_cca_positions])
            avg_prev = prev_distances.mean()
            max_prev = prev_distances.max()
            avg_reduction = avg_prev - avg_dist
            scaled_avg = avg_reduction * (500.0 / self.grid_size)
            reward += avg_progress_coeff * np.clip(scaled_avg, -2.0, 2.0)

            max_reduction = max_prev - max_dist
            scaled_max = max_reduction * (500.0 / self.grid_size)
            reward += max_progress_coeff * np.clip(scaled_max, -2.0, 2.0)

        # Variance penalty on distances
        if self.num_cca > 1:
            variance = np.var(distances)
            norm_variance = variance / (self.grid_size ** 2)
            reward += -variance_penalty_coeff * norm_variance * 100

        # Minimum distance penalty
        reward += -min_dist_penalty_coeff * min_dist

        # Partial capture and bonus rewards
        num_capturing = sum(dist < self.capture_radius for dist in distances)
        if num_capturing > 0:
            capture_percentage = num_capturing / self.num_cca
            reward += partial_capture_coeff * capture_percentage * capture_bonus
            if num_capturing == self.num_cca:
                reward += all_agents_capture_bonus

        # Surround formation reward
        if self.num_cca > 1 and avg_dist < self.grid_size / 4:
            approach_angles = []
            for cca in self.cca_objs:
                vector = self.foxtrot_obj.position - cca.position
                if np.linalg.norm(vector) > 0:
                    approach_angles.append(np.arctan2(vector[1], vector[0]))
            if approach_angles:
                angle_diffs = []
                for i in range(len(approach_angles)):
                    for j in range(i + 1, len(approach_angles)):
                        diff = min(abs(approach_angles[i] - approach_angles[j]),
                                   2 * np.pi - abs(approach_angles[i] - approach_angles[j]))
                        angle_diffs.append(diff)
                if angle_diffs:
                    ideal_diff = 2 * np.pi / self.num_cca
                    avg_angle_diff = np.mean(angle_diffs)
                    quality = max(0, min(1, avg_angle_diff / ideal_diff))
                    reward += surround_coeff * quality

        # Laggard boost for agents that are trailing
        if self.previous_cca_positions is not None and self.num_cca > 1:
            percentile_75 = np.percentile(distances, 75)
            laggard_reward = 0
            for i, dist in enumerate(distances):
                if dist >= percentile_75:
                    prev = np.linalg.norm(self.previous_cca_positions[i] - self.foxtrot_obj.position)
                    prog = prev - dist
                    if prog > 0:
                        scaled = prog * (500.0 / self.grid_size)
                        laggard_reward += laggard_coeff * np.clip(scaled, 0, 1.0)
            reward += laggard_reward

        # Add time and energy penalties
        reward += time_penalty
        if actions is not None:
            energy_use = np.sum(actions**2) / (step_size**2 * self.num_cca)
            reward += -energy_penalty_coeff * energy_use * 100

        # Collision penalty based on average separation
        if avg_sep < self.cca_collision_radius:
            reward += collision_penalty

        # Capture bonus if any agent is within capture radius
        if min_dist < self.capture_radius:
            reward += capture_bonus

        return reward

    def _calculate_collision_penalty(self, current_cca_positions):
        """Penalty for CCA collisions."""
        penalty = 0.0
        if self.num_cca > 1:
            for i in range(self.num_cca):
                for j in range(i + 1, self.num_cca):
                    sep = np.linalg.norm(current_cca_positions[i] - current_cca_positions[j])
                    if sep < self.cca_collision_radius:
                        penalty -= 100.0
        return penalty
    
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
    env = PPOEnv(grid_size=500, num_cca=8)
    #env = Monitor(env, log_dir) # Wrap with Monitor BEFORE DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    # IMPORTANT: Normalize observations. May need adjustment for placeholder value (-1).
    # Consider normalizing reward as well.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=500.0) # Increased clip_obs range


    # --- Define LMA Hyperparameters ---
    # ** TUNE THESE CAREFULLY for L=200 **
    lma_kwargs = dict(
        embed_dim=256,          # d0: Increased initial embedding?
        num_heads_stacking=16,   # nh: (128 % 8 == 0) -> dk=16
        target_l_new=20,        # Target L_new (L=200) -> find divisor for 200*128=25600
                                # 50 is divisor. L_new=50 -> C_new=512
        d_new=128,               # d_new: Latent dimension
        num_heads_latent=16,     # Latent heads (64 % 8 == 0) -> latent_dk=8
        ff_latent_hidden=128*6,   # Latent MLP hidden (e.g., 2*d_new)
        num_lma_layers=6,       # More layers for longer sequence and harder task
        seq_len=30,            # Initial Sequience length (L=100)
        dropout=0.1,
        bias=True
    )

    # --- Define Policy Keyword Arguments ---
    # Input to MLP head is L_new * d_new = 50 * 64 = 3200
    policy_kwargs = dict(
        features_extractor_class=LMAFeaturesExtractor,
        features_extractor_kwargs=lma_kwargs,
        # Adjust net_arch based on new feature dim (3200) - maybe needs more capacity
        net_arch=dict(pi=[512,512, 256, 128], vf=[512, 512, 256, 64]) # Example larger arch
    )

    # --- Define PPO Model ---
    model_class = PPO
    model_specific_kwargs = {}

    model = model_class(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        normalize_advantage=True,
        learning_rate=linear_schedule(initial_value= 0.0003), # Possibly lower LR for harder task
        n_steps=800,             # Increase steps for more data per update
        batch_size=200,           # Increase batch size
        n_epochs=10,
        gamma=0.9,               # Standard discount factor
        gae_lambda=0.85,           # Standard GAE factor
        clip_range=0.2,           # Standard PPO clip range
        ent_coef=0.002,           # Small entropy encourages exploration slightly
        vf_coef=0.7,              # Value function coefficient
        max_grad_norm=0.5,
        tensorboard_log= "./TensorBoardLogs/",
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