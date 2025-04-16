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
import time
import random # For curriculum selection
from classes import *

# CURRENTLY A BUG - ALL ACTIONS ARE THE SAME???


# Internal Module Imports - Assumes classes.py is updated
try:
    # Import necessary classes - LMAFeaturesExtractor, Transformer, CCA, Foxtrot
    from classes import *
except ImportError as e:
    print(f"Error importing from classes.py: {e}")
    print("Please ensure classes.py is updated and in the same directory or your PYTHONPATH.")
    exit()

# Import globals or set defaults
try:
    import globals
    REWARD_DEBUG = getattr(globals, 'REWARD_DEBUG', False)
    POSITIONAL_DEBUG = getattr(globals, 'POSITIONAL_DEBUG', False)
    DEFAULT_GRID_SIZE = getattr(globals, 'grid_size', 500)
    DEFAULT_STEP_SIZE = getattr(globals, 'step_size', 10)
    globals.STATIONARY_FOXTROT = False
    globals.RECTANGULAR_FOXTROT = True # Foxtrot moves in cube pattern
    globals.RAND_POS = True
    globals.FIXED_POS = False
    globals.PROXIMITY_CCA = True
    globals.RAND_FIXED_CCA = False
except ImportError:
    print("WARNING!: 'globals.py' not found or not fully configured. Using defaults.")
    REWARD_DEBUG = False
    POSITIONAL_DEBUG = False
    DEFAULT_GRID_SIZE = 500
    DEFAULT_STEP_SIZE = 10
    class MockGlobals:
        STATIONARY_FOXTROT = False
        RECTANGULAR_FOXTROT = True
        RAND_POS = True
        FIXED_POS = False
        PROXIMITY_CCA = True
        RAND_FIXED_CCA = False
    globals = MockGlobals()

# --- REMOVED Foxtrot Patching Code ---
# Assumes move_cube logic is now INSIDE the Foxtrot class in classes.py

#===============================================
# Swarm PPO Environment with Curriculum Input & Potential Reward
#===============================================
class PPOSwarmCurriculumEnv(gym.Env):
    """
    Env for N CCAs, full observability, L history.
    Objective: Maintain MULTIPLE formations relative to a MOVING Foxtrot.
    Formation type is part of the observation (one-hot encoded).
    Uses potential-based reward shaping.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}
    SUPPORTED_FORMATIONS = ['square', 'circle', 'line', 'wedge']

    def __init__(self, grid_size=DEFAULT_GRID_SIZE, num_cca=4, history_len=31,
                 step_size_env=DEFAULT_STEP_SIZE, formation_scale=50.0,
                 initial_formation=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_cca = num_cca
        self.history_len = history_len
        self.step_size = step_size_env
        self.formation_scale = formation_scale
        self.cca_collision_radius = 5.0 # For collision penalty
        self.max_steps = 400

        # Curriculum State
        self.formation_map = {name: i for i, name in enumerate(self.SUPPORTED_FORMATIONS)}
        self.num_formations = len(self.SUPPORTED_FORMATIONS)
        if initial_formation and initial_formation in self.formation_map:
             self.current_formation_type = initial_formation
        else:
             self.current_formation_type = random.choice(self.SUPPORTED_FORMATIONS)
        self.current_formation_idx = self.formation_map[self.current_formation_type]
        self.current_formation_onehot = np.zeros(self.num_formations, dtype=np.float32)
        self.current_formation_onehot[self.current_formation_idx] = 1.0
        self.target_formation_offsets = self._get_formation_offsets(self.current_formation_type)

        # Env State
        self.current_step = 0
        self.foxtrot_obj = None
        self.cca_objs = []
        self.previous_cca_positions = None
        self.current_potential = [0.0] * self.num_cca # For potential-based reward
        self.previous_potential = [0.0] * self.num_cca # For potential-based reward
        self.previous_foxtrot_pos = None

        # Action Space
        self.action_space = spaces.Box(low=-self.step_size, high=self.step_size, shape=(self.num_cca, 3), dtype=np.float32)

        # Histories
        self.action_history = None; self.cca_history = None; self.foxtrot_history = None
        self.formation_cmd_history = None

        # Observation Space
        self.obs_per_step_dynamic = (3 + 3) * self.num_cca + 3
        self.obs_per_step_cmd = self.num_formations
        self.observation_dim_per_step = self.obs_per_step_dynamic + self.obs_per_step_cmd
        self.observation_shape_total = self.history_len * self.observation_dim_per_step

        # Observation Bounds
        high_pos = float(self.grid_size); high_act = float(self.step_size); high_cmd = 1.0
        low_pos = 0.0; low_act = -float(self.step_size); low_cmd = 0.0
        low_bounds_step, high_bounds_step = [], []
        for _ in range(self.num_cca): # CCA Pos/Act
            low_bounds_step.extend([low_pos]*3); high_bounds_step.extend([high_pos]*3)
            low_bounds_step.extend([low_act]*3); high_bounds_step.extend([high_act]*3)
        low_bounds_step.extend([low_pos]*3); high_bounds_step.extend([high_pos]*3) # Foxtrot Pos
        low_bounds_step.extend([low_cmd]*self.obs_per_step_cmd); high_bounds_step.extend([high_cmd]*self.obs_per_step_cmd) # Command
        low_bounds = np.tile(low_bounds_step, self.history_len); high_bounds = np.tile(high_bounds_step, self.history_len)

        self.observation_space = spaces.Box(
            low=low_bounds.astype(np.float32), high=high_bounds.astype(np.float32),
            shape=(self.observation_shape_total,), dtype=np.float32,
        )

        print(f"\nPPOSwarmCurriculumEnv Initialized ({self.num_cca} Agents, L={self.history_len}):")
        print(f"  Supported Formations: {self.SUPPORTED_FORMATIONS}")
        print(f"  Grid: {self.grid_size}, Step Size: {self.step_size}, Scale: {self.formation_scale}")
        print(f"  Obs Dim Per Step: {self.observation_dim_per_step} (Dyn: {self.obs_per_step_dynamic}, Cmd: {self.obs_per_step_cmd})")
        print(f"  Total Obs Dim: {self.observation_shape_total}")

    def _get_formation_offsets(self, formation_type):
        # (Identical logic as before, calculates offsets for the given type)
        offsets = np.zeros((self.num_cca, 3)); scale = self.formation_scale
        if self.num_cca == 0: return offsets
        if formation_type == 'line':
            start_x = -scale * (self.num_cca - 1) / 2.0
            for i in range(self.num_cca): offsets[i] = [start_x + i * scale, 0, 0]
        elif formation_type == 'square':
            side_len = int(np.ceil(np.sqrt(self.num_cca))); idx = 0
            for r in range(side_len):
                for c in range(side_len):
                    if idx < self.num_cca: x=(c-(side_len-1)/2.0)*scale; y=(r-(side_len-1)/2.0)*scale; offsets[idx]=[x,y,0]; idx+=1
        elif formation_type == 'circle':
            radius = scale
            for i in range(self.num_cca): angle=2*np.pi*i/self.num_cca; offsets[i]=[radius*np.cos(angle), radius*np.sin(angle), 0]
        elif formation_type == 'wedge':
             angle_spread = np.pi / 3
             for i in range(self.num_cca):
                 frac = (i - (self.num_cca - 1)/2.0) / max(1, self.num_cca-1) if self.num_cca > 1 else 0
                 angle = frac * angle_spread; dist = scale * (1 + abs(frac) * 0.5)
                 offsets[i] = [dist * np.cos(angle), dist * np.sin(angle), 0]
        else: raise ValueError(f"Unknown formation type requested: {formation_type}")
        return offsets.astype(np.float32)

    def set_active_formation(self, formation_type):
        """Externally sets the formation for the *next* episode (used in eval)."""
        if formation_type not in self.formation_map:
            formation_type = random.choice(self.SUPPORTED_FORMATIONS)
            print(f"Warning: Formation '{formation_type}' not supported. Using random '{formation_type}'.")
        self.current_formation_type = formation_type
        self.current_formation_idx = self.formation_map[self.current_formation_type]
        self.current_formation_onehot = np.zeros(self.num_formations, dtype=np.float32)
        self.current_formation_onehot[self.current_formation_idx] = 1.0
        self.target_formation_offsets = self._get_formation_offsets(self.current_formation_type)
        print(f"  Env: Active formation set to '{self.current_formation_type}' for next reset.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_cca_positions = None
        self.current_potential = [0.0] * self.num_cca # Reset potential for new episode
        self.previous_potential = [0.0] * self.num_cca # Reset potential for new episode
        self.previous_foxtrot_pos = None

        # Curriculum Logic: Select formation unless 'keep_formation' is set for eval
        if not options or not options.get("keep_formation", False):
             self.current_formation_type = self.np_random.choice(self.SUPPORTED_FORMATIONS)
             self.current_formation_idx = self.formation_map[self.current_formation_type]
             self.current_formation_onehot.fill(0.0); self.current_formation_onehot[self.current_formation_idx] = 1.0
             self.target_formation_offsets = self._get_formation_offsets(self.current_formation_type)

        print(f"--- ENV RESET (Formation: {self.current_formation_type}) ---")

        # Reset Foxtrot - Assumes Foxtrot class handles internal state reset in set_position
        initial_foxtrot_pos = self.np_random.integers(self.grid_size*0.4, self.grid_size*0.6, size=3).astype(float)
        if self.foxtrot_obj is None: self.foxtrot_obj = Foxtrot("Foxtrot_0", initial_foxtrot_pos)
        else: self.foxtrot_obj.set_position(initial_foxtrot_pos)

        # Reset CCAs - Use the wider spawn range from user's script
        self.cca_objs = []
        target_positions = self.foxtrot_obj.position + self.target_formation_offsets
        for i in range(self.num_cca):
            # Use the wider spawn range
            if globals.PROXIMITY_CCA:
                start_pos = target_positions[i] + self.np_random.uniform(-30, 30, size=3)
                start_pos = np.clip(start_pos, 0, self.grid_size - 1)
                self.cca_objs.append(CCA(f"CCA_{i}", start_pos))
            else:
                start_pos = target_positions[i] + self.np_random.uniform(-300, 300, size=3)
                start_pos = np.clip(start_pos, 0, self.grid_size - 1)
                self.cca_objs.append(CCA(f"CCA_{i}", start_pos))

        # Reset Histories
        self.action_history = [np.zeros((self.history_len, 3), dtype=np.float32) for _ in range(self.num_cca)]
        self.cca_history = [np.tile(cca.position, (self.history_len, 1)) for cca in self.cca_objs]
        self.foxtrot_history = np.tile(self.foxtrot_obj.position, (self.history_len, 1))
        self.formation_cmd_history = np.tile(self.current_formation_onehot, (self.history_len, 1))

        self.previous_cca_positions = [cca.position.copy() for cca in self.cca_objs]

        if POSITIONAL_DEBUG:
            print(f"{self.foxtrot_obj.name} @ {self.foxtrot_obj.position.round(1)}\n" +
                  "\n".join([f"{c.name} @ {c.position.round(1)} (Target: {target_positions[i].round(1)})" for i, c in enumerate(self.cca_objs)]) +
                  "\n--------------")

        observation = self._get_observation()
        info = self.get_current_metrics(); info["current_formation"] = self.current_formation_type
        return observation, info

    def step(self, actions):
        self.current_step += 1
        actions = np.array(actions, dtype=np.float32).reshape(self.num_cca, 3)
        clipped_actions = np.clip(actions, -self.step_size, self.step_size)

        # Store current positions before movement for reward calculation
        self.previous_cca_positions = [cca.position.copy() for cca in self.cca_objs] # Store before move\
        self.previous_foxtrot_pos = self.foxtrot_obj.position.copy() # Store before move

        # --- Move CCAs ---
        current_cca_positions = []
        for i in range(self.num_cca):
            self.action_history[i]=np.roll(self.action_history[i],shift=-1,axis=0); self.action_history[i][-1]=clipped_actions[i]
            self.cca_objs[i].move(clipped_actions[i])
            self.cca_objs[i].position = np.clip(self.cca_objs[i].position, 0, self.grid_size - 1)
            self.cca_history[i]=np.roll(self.cca_history[i],shift=-1,axis=0); self.cca_history[i][-1]=self.cca_objs[i].position
            current_cca_positions.append(self.cca_objs[i].position.copy())

         # --- Calculate Reward using Potential Shaping ---
        reward = self._calculate_swarm_reward_potential(clipped_actions, current_cca_positions, self.foxtrot_obj.position)
        
        # --- Move Foxtrot ---
        if globals.RECTANGULAR_FOXTROT:
            self.foxtrot_obj.move_cube(self.grid_size, self.step_size)
        current_foxtrot_pos = self.foxtrot_obj.position.copy()
        
        # --- Update Histories ---
        self.foxtrot_history = np.roll(self.foxtrot_history, shift=-1, axis=0); self.foxtrot_history[-1] = current_foxtrot_pos
        self.formation_cmd_history = np.roll(self.formation_cmd_history, shift=-1, axis=0); self.formation_cmd_history[-1] = self.current_formation_onehot

        # Termination/Truncation
        done = False
        truncated = False

        if self.current_step == self.max_steps:
            done = truncated = True

        observation = self._get_observation()
        info = self.get_current_metrics(); info["current_formation"] = self.current_formation_type

        return observation, reward, done, truncated, info

    def _get_observation(self):
        # (Identical logic as before, concatenates all history parts including command)
        state_components = []
        for t in range(self.history_len):
            for i in range(self.num_cca): state_components.append(self.cca_history[i][t]); state_components.append(self.action_history[i][t])
            state_components.append(self.foxtrot_history[t])
            state_components.append(self.formation_cmd_history[t])
        obs_flat = np.concatenate(state_components).astype(np.float32)
        if obs_flat.shape != (self.observation_shape_total,):
             calc_expected = sum(comp.size for comp in state_components); raise ValueError(f"Obs shape mismatch! Exp ({self.observation_shape_total},), Got {obs_flat.shape}. Components sum to {calc_expected}.")
        return obs_flat

    def _calculate_potential(self, cca_positions_list, foxtrot_pos):
        """ Helper to calculate the potential function value. """
        # Potential is the negative average distance to the target formation slot
        target_positions = foxtrot_pos + self.target_formation_offsets
        formation_errors = []
        for i in range(self.num_cca):
            # Ensure positions are numpy arrays for linalg.norm
            cca_pos = np.asarray(cca_positions_list[i])
            target_pos = np.asarray(target_positions[i])
            error = np.linalg.norm(cca_pos - target_pos)
            formation_errors.append(error)

        if not formation_errors: return 0.0
        avg_formation_error = np.mean(formation_errors)
        # Negative error: higher potential means lower error (closer to target)
        return -avg_formation_error

    def _calculate_swarm_reward_potential(self, actions, current_cca_positions, current_foxtrot_pos):
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
        capture_radius = 50.0    # Radius for capture bonus
        
        target_positions = current_foxtrot_pos + self.target_formation_offsets
        
        # Define the potential function
        def potential(pos):
            return float(-np.mean([np.linalg.norm(pos - self.foxtrot_obj.position)]))

        potential_reward = 0
        for i in range(self.num_cca):
        # Current and previous potential
            self.current_potential[i] = potential(target_positions[i])
            # Potential-based shaping (identical to reference)
            shaped_reward = gamma * (self.current_potential[i] - self.previous_potential[i])
            # Update previous potential for next step
            self.previous_potential[i] = self.current_potential[i]
            potential_reward += shaped_reward

        if self.num_cca > 0:
            potential_reward /= self.num_cca
            reward += potential_reward
            
            
        # Progress-Based Reward (simplified to match reference)
        total_progress = 0.0
        for i in range(self.num_cca):
            # Calculate progress toward target position (simpler)
            current_distance = np.linalg.norm((current_cca_positions[i]) - (target_positions[i]))
            prev_distance = np.linalg.norm((self.previous_cca_positions[i]) - 
                                        (target_positions[i]))
            
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
            distances_str = ", ".join([f"{np.linalg.norm(np.asarray(current_cca_positions[i]) - np.asarray(target_positions[i])):.2f}" for i in range(self.num_cca)])
            print(f"Distances to targets: [{distances_str}], Raw Reward: {reward}, Progress Reward: {total_progress}, Potential Reward: {potential_reward}")
        
        return float(reward)


    def get_current_metrics(self):
        # (Identical logic as before, calculates metrics for the current formation)
        if not self.cca_objs or self.foxtrot_obj is None: return {"avg_formation_error": float('inf'), "avg_separation": 0.0, "min_separation": 0.0, "collisions": 0}
        current_cca_pos = np.array([cca.position for cca in self.cca_objs]); current_foxtrot_pos = self.foxtrot_obj.position
        target_positions = current_foxtrot_pos + self.target_formation_offsets
        errors = [np.linalg.norm(current_cca_pos[i] - target_positions[i]) for i in range(self.num_cca)]; avg_error = np.mean(errors) if errors else 0.0
        seps = []; colls = 0
        if self.num_cca > 1:
            for i in range(self.num_cca):
                for j in range(i + 1, self.num_cca):
                    sep = np.linalg.norm(current_cca_pos[i] - current_cca_pos[j])
                    seps.append(sep)
                    if sep < self.cca_collision_radius:
                        colls += 1
            avg_sep = np.mean(seps) if seps else 0.0; min_sep = min(seps) if seps else 0.0
        else: avg_sep = 0.0; min_sep = 0.0
        return {"avg_formation_error": float(avg_error), "avg_separation": float(avg_sep), "min_separation": float(min_sep), "collisions": colls}

    def render(self):
         # (Identical logic as before)
         print(f"--- Step {self.current_step} (Target Formation: {self.current_formation_type}) ---")
         print(f"Foxtrot: {self.foxtrot_obj.position.round(1)}")
         metrics = self.get_current_metrics(); print(f"Avg Formation Error: {metrics['avg_formation_error']:.2f}"); print(f"Avg/Min Separation: {metrics['avg_separation']:.2f} / {metrics['min_separation']:.2f}"); print(f"Collisions: {metrics['collisions']}")
         targets = self.foxtrot_obj.position + self.target_formation_offsets
         for i, cca in enumerate(self.cca_objs): print(f"  {cca.name}: {cca.position.round(1)} (Target: {targets[i].round(1)})")

    def close(self): pass

#===============================================
# Main Training Block
#===============================================
if __name__ == "__main__":

    def linear_schedule(initial_value: float):
        def schedule(progress_remaining: float): return progress_remaining * initial_value
        return schedule

    # --- Environment Configuration ---
    GRID_SIZE = DEFAULT_GRID_SIZE
    STEP_SIZE = DEFAULT_STEP_SIZE
    NUM_CCA = 4
    HISTORY_LEN = 10
    FORMATION_SCALE = 40.0

    # --- Create and wrap the environment ---
    env = PPOSwarmCurriculumEnv(grid_size=GRID_SIZE, num_cca=NUM_CCA, history_len=HISTORY_LEN,
                                step_size_env=STEP_SIZE, formation_scale=FORMATION_SCALE)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=float(GRID_SIZE), gamma=0.99) # Keep gamma high

    # --- Feature Extractor Selection ---
    config = input("Choose Model Config: (LMA/MHA/MHA_Lite): ")
    if config == "LMA":
        print("Using LMA configuration.")
        FeatureExtractor = LMAFeaturesExtractor
        # Use user's specific LMA kwargs
        kwargs = dict(
            seq_len=HISTORY_LEN, embed_dim=1024, num_heads_stacking=32,
            target_l_new=int(HISTORY_LEN/2), # Ensure integer division if needed
            d_new=512, num_heads_latent=32, ff_latent_hidden=512*6,
            num_lma_layers=6, dropout=0.2, bias=True
        )
    elif config == "MHA" or config == "MHA_Lite": # Combine MHA/MHA_Lite logic
         print(f"Using {config} configuration (Transformer).")
         FeatureExtractor = Transformer
         if config == "MHA":
             kwargs=dict(embed_dim=128, num_heads=8, ff_hidden=128*4, num_layers=4, seq_len=HISTORY_LEN)
         else: # MHA_Lite
             kwargs=dict(embed_dim=64, num_heads=8, ff_hidden=64*4, num_layers=4, seq_len=HISTORY_LEN)
    else:
        print("Invalid configuration selected. Exiting.")
        exit()
    print(f"Feature Extractor Kwargs: {kwargs}")


    # --- Define Policy Keyword Arguments ---
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=kwargs,
        # Use user's MLP head sizes
        net_arch=dict(pi=[512,512,256, 256], vf=[512,512,512,256, 128])
    )

    # --- Define PPO Model ---
    model_class = PPO

    # Use user's PPO hyperparameters
    model = model_class(
        policy="MlpPolicy", policy_kwargs=policy_kwargs, env=vec_env, verbose=1,
        normalize_advantage=True, learning_rate=linear_schedule(0.00009),
        n_steps=600, batch_size=100, n_epochs=10,
        gamma=0.9, gae_lambda=0.9, clip_range=0.4,
        ent_coef=0.002, vf_coef=0.85, max_grad_norm=0.5,
        tensorboard_log= "./TensorBoardLogs"
    )

    # --- Optional: Display Model Summary ---
    try:
        from torchinfo import summary
        if hasattr(vec_env, 'observation_space') and vec_env.observation_space is not None:
             # Need dummy batch size
             summary(model.policy, input_size=(1, *vec_env.observation_space.shape))
        else: print("Warning: vec_env.observation_space not available for summary.")
    except ImportError: print("torchinfo not found.")
    except Exception as e: print(f"Error getting model summary: {e}")


    proceed = input("Review model summary and config. Continue with training? (y/n): ")
    if proceed.lower() != 'y': print("Training aborted."); vec_env.close(); exit()

    # --- Training ---
    print("\nStarting Training (Swarm Curriculum)...")
    
    try:
        model.learn(total_timesteps=400_000)
        #globals.PROXIMITY_CCA = False
        #model.learn(total_timesteps=100_000) #You should implement ciriculum learning within the enviornment itself
    except KeyboardInterrupt: print("\nTraining interrupted.")
    except Exception as e: print(f"\nTraining error: {e}"); import traceback; traceback.print_exc()

    # --- Save Model and Environment Wrapper ---
    save_dir = f"./trained_swarm_curriculum_{config}/" # Add config to save dir
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "swarm_curriculum_ppo_model")
    vec_norm_path = os.path.join(save_dir, "swarm_curriculum_vecnormalize.pkl")
    try:
        print(f"\nSaving model to {model_path}"); model.save(model_path)
        print(f"Saving VecNormalize stats to {vec_norm_path}"); vec_env.save(vec_norm_path)
        print("Save successful.")
    except Exception as e: print(f"\nSave error: {e}"); traceback.print_exc()


    # --- Evaluation per Formation Type ---
    print("\n--- Starting Evaluation (Per Formation) ---")
    eval_episodes_per_type = 5
    all_eval_results = {}

    try:
        # Create lambda for eval env creation
        eval_env_lambda = lambda: PPOSwarmCurriculumEnv(grid_size=GRID_SIZE, num_cca=NUM_CCA, history_len=HISTORY_LEN, step_size_env=STEP_SIZE, formation_scale=FORMATION_SCALE)
        eval_vec_env = DummyVecEnv([eval_env_lambda])

        print(f"Loading VecNormalize stats from: {vec_norm_path}")
        eval_vec_env = VecNormalize.load(vec_norm_path, eval_vec_env)
        eval_vec_env.training = False; eval_vec_env.norm_reward = False

        print(f"Loading model from: {model_path}")
        eval_model = PPO.load(model_path, env=eval_vec_env)

        for form_type in PPOSwarmCurriculumEnv.SUPPORTED_FORMATIONS:
            print(f"\n--- Evaluating Formation: {form_type} ---")
            # Initialize results storage for this formation type
            results_key = form_type
            all_eval_results[results_key] = {"rewards": [], "avg_formation_error": [], "avg_separation": [], "min_separation": [], "collisions": []}

            eval_vec_env.env_method('set_active_formation', form_type)
            obs = eval_vec_env.reset() # Reset *after* setting formation

            for episode in range(eval_episodes_per_type):
                done=False; cumulative_reward=0.0; step=0; ep_metrics=[]
                while not done:
                    action, _ = eval_model.predict(obs, deterministic=True)
                    obs, _, done_vec, info_vec = eval_vec_env.step(action) # Use dummy vars for reward
                    actual_reward = eval_vec_env.get_original_reward()[0]
                    info = info_vec[0]; done = done_vec[0]
                    cumulative_reward += actual_reward
                    if "avg_formation_error" in info: ep_metrics.append(info)
                    step += 1
                    if step >= env.max_steps: done = True # Manual truncation

                    if done:
                        print(f"  Ep {episode+1}/{eval_episodes_per_type}: Steps={step}, Reward={cumulative_reward:.2f}")
                        all_eval_results[results_key]["rewards"].append(cumulative_reward)
                        if ep_metrics:
                             all_eval_results[results_key]["avg_formation_error"].append(np.mean([m['avg_formation_error'] for m in ep_metrics]))
                             all_eval_results[results_key]["avg_separation"].append(np.mean([m['avg_separation'] for m in ep_metrics]))
                             all_eval_results[results_key]["min_separation"].append(min([m['min_separation'] for m in ep_metrics]))
                             all_eval_results[results_key]["collisions"].append(sum([m.get('collisions', 0) for m in ep_metrics])) # Use .get for safety
                        break # Exit while loop

        eval_vec_env.close()

        # Print summary results
        print("\n--- Evaluation Summary ---")
        for form_type, results in all_eval_results.items():
            print(f"\nFormation: {form_type}")
            if results["rewards"]:
                print(f"  Avg Reward: {np.mean(results['rewards']):.2f} +/- {np.std(results['rewards']):.2f}")
                print(f"  Avg Formation Error: {np.mean(results['avg_formation_error']):.2f} +/- {np.std(results['avg_formation_error']):.2f}")
                print(f"  Avg Separation: {np.mean(results['avg_separation']):.2f} +/- {np.std(results['avg_separation']):.2f}")
                print(f"  Avg Total Collisions: {np.mean(results['collisions']):.2f} +/- {np.std(results['collisions']):.2f}")
            else: print("  No metrics collected.")

    except FileNotFoundError: print(f"\nEval Error: File not found.\nPaths: {model_path}, {vec_norm_path}")
    except Exception as e: print(f"\nEvaluation error: {e}"); traceback.print_exc()

    # --- Cleanup ---
    try: vec_env.close()
    except Exception as e: print(f"Error closing training env: {e}")
    print("\nTraining and evaluation script finished.")