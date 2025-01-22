import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from math import pow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# Internal Module Imports
from classes import *
from globals import (
    grid_size,
    num_cca,
    step_size,
    STATIONARY_FOXTROT,
    RECTANGULAR_FOXTROT,
    COMPLEX_REWARD, 
    BASIC_REWARD,
    FIXED_POS,
    RAND_POS,
    RAND_FIXED_CCA,
    spawn_range
)

DEBUG = True
REWARD_DEBUG = True
POSITIONAL_DEBUG = True

class PPOEnv(gym.Env):
    """Custom PPO Environment for controlling CCA objects."""
    def __init__(self, grid_size=500, num_cca=1):
        super(PPOEnv, self).__init__()
        self.grid_size = grid_size
        self.num_cca = num_cca
        self.current_step = 0
        self.max_steps = 1000 # Epsiode Length in the simulation
        self.cube_state = {}

        # Action space: Continuous movement in 3D space
        self.action_space = spaces.Box(
            low=-step_size, high=step_size, shape=(self.num_cca, 3), dtype=np.float32
        )

        # Observation space: Includes last 5 positions for CCAs and Foxtrot
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(3 * self.num_cca * 6 + 3 * 6,),  # Add 6 sets of 3D positions for Foxtrot
            dtype=np.float32
        )

        # Initialize Foxtrot positions
        if RAND_POS:
            self.foxtrot_position = np.random.randint(200, 301, size=3)
        elif FIXED_POS:
            self.foxtrot_position = np.array([250,250,250])

        # Initialize CCA positions (randomized in the reset function)
        if RAND_FIXED_CCA:
            self.cca_positions = [
                np.clip(
                    self.foxtrot_position + np.random.randint(-spawn_range, spawn_range + 1, size=3),
                    0, self.grid_size - 1
                ) for _ in range(self.num_cca)
            ]
        
        else:
            self.cca_positions = [
            np.random.randint(0, self.grid_size, size=3) for _ in range(self.num_cca)
            ]

        # Initialize history for last 5 positions
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))  # 6 rows for 5 past + current position

    def reset(self, seed=None, **kwargs):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)  # Properly seed the environment

        self.cube_state = {}
        self.current_step = 0
        
        # Create CCA Positions
        if RAND_FIXED_CCA:
            self.cca_positions = [
                np.clip(
                    self.foxtrot_position + np.random.randint(-spawn_range, spawn_range + 1, size=3),
                    0, self.grid_size - 1
                ) for _ in range(self.num_cca)
            ]
        
        else:
            self.cca_positions = [
            np.random.randint(0, self.grid_size, size=3) for _ in range(self.num_cca)
            ]

        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]

        # Randomize Foxtrot position on the cube path
        side_length = 200  # Length of each cube edge
        half_side = side_length // 2
        center = np.array([250, 250, 250])  # Center of the cube
        cube_vertices = [
            center + np.array([half_side, half_side, half_side]),
            center + np.array([-half_side, half_side, half_side]),
            center + np.array([-half_side, -half_side, half_side]),
            center + np.array([half_side, -half_side, half_side]),
            center + np.array([half_side, -half_side, -half_side]),
            center + np.array([half_side, half_side, -half_side]),
            center + np.array([-half_side, half_side, -half_side]),
            center + np.array([-half_side, -half_side, -half_side]),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
            (0, 5), (1, 6), (2, 7), (3, 4)   # Vertical edges
        ]

        if RECTANGULAR_FOXTROT:
            # Randomly select an edge and a point along the edge
            random_edge_index = np.random.choice(len(edges))
            edge_start, edge_end = edges[random_edge_index]
            random_progress = np.random.uniform(0, 1)  # Random progress along the edge
            self.foxtrot_position = (1 - random_progress) * cube_vertices[edge_start] + random_progress * cube_vertices[edge_end]
            self.foxtrot_position = np.round(self.foxtrot_position).astype(int)

        if STATIONARY_FOXTROT:
            if RAND_POS:
                self.foxtrot_position = np.random.randint(200, 301, size=3)
            elif FIXED_POS:
                self.foxtrot_position = np.array([250,250,250])

        # Reset Foxtrot history
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))

        if POSITIONAL_DEBUG:
            print(f"Foxtrot spawned at {self.foxtrot_position}")
            for i, pos in enumerate(self.cca_positions):
                print(f"CCA {i} spawned at {pos}")
                
        obs = self._get_observation()
        info = {}  # Add an empty dictionary for compatibility
        return obs, info

    def step(self, actions):
        # Convert actions to a NumPy array in case it's not already
        actions = np.array(actions, dtype=np.float32)

        # If the shape is (num_cca * 3,), reshape to (num_cca, 3)
        # This ensures we handle both the single- and multi-CCA cases correctly
        if actions.ndim == 1 and actions.size == self.num_cca * 3:
            actions = actions.reshape(self.num_cca, 3)
        elif actions.ndim == 2:
            # Already shaped (num_cca, 3); no change needed
            pass
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")

        # Now 'actions' has shape (num_cca, 3), so your loop works as intended
        for i, action in enumerate(actions):
            movement_vector = self._decode_action(action)
            self.cca_positions[i] += movement_vector.astype(int)
            self.cca_positions[i] = np.clip(self.cca_positions[i], 0, self.grid_size - 1)

            # Update CCA history
            self.cca_history[i] = np.roll(self.cca_history[i], shift=-1, axis=0)
            self.cca_history[i][-1] = self.cca_positions[i]

        # Update Foxtrot position and history
            
        if RECTANGULAR_FOXTROT:
            self.foxtrot_position = foxtrot_movement_fn_cube(self.foxtrot_position, self.cube_state)
            self.foxtrot_position = np.clip(self.foxtrot_position, 0, self.grid_size - 1)

        # Update Foxtrot history
        self.foxtrot_history = np.roll(self.foxtrot_history, shift=-1, axis=0)
        self.foxtrot_history[-1] = self.foxtrot_position

        # Calculate reward
        reward = self._calculate_reward()

        # Check success condition for ANY of the CCAs:
        done_success = False
        for i, pos in enumerate(self.cca_positions):
            distance = np.linalg.norm(pos - self.foxtrot_position)
            if distance < 5.0:  # same capture_radius
                done_success = True
                break

        # Time-limit done
        self.current_step += 1
        truncated = (self.current_step >= self.max_steps)

        # If agent succeeded, terminate the episode
        terminated = done_success  # or keep it False if you want continuing episodes
        
        # Return standard Gymnasium step
        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        # We have self.cca_history[i] shape (6, 3) for 6 time steps
        # and self.foxtrot_history shape (6, 3)
        
        state = []
        for t in range(6):  # for each time step
            # If you had multiple CCAs, you’d loop i in range(num_cca), etc.
            state.extend(self.cca_history[0][t])    # shape (3,)
            state.extend(self.foxtrot_history[t])   # shape (3,)

        return np.array(state, dtype=np.float32)

    def _decode_action(self, action):
        """Clamp action values to ensure they stay within bounds."""
        return np.clip(action, -step_size, step_size)

    def _calculate_reward(self):
        """
        Calculate reward based on the current reward flag settings.
        """
        if COMPLEX_REWARD:
            # Complex reward logic
            reward = 0.0
            alpha = 300.0
            beta = 0.1  
            capture_bonus = 1000.0
            capture_radius = 5.0
            alpha_capture_radius = 20
            beta_capture_radius = 50

            for i, pos in enumerate(self.cca_positions):
                distances = [
                    np.linalg.norm(history_pos - self.foxtrot_position)
                    for history_pos in self.cca_history[i][-3:]  # Use last 3 positions
                ]
                avg_improvement = np.mean([
                    (distances[j] - distances[j+1]) / distances[j] if distances[j] > 0 else 0
                    for j in range(len(distances) - 1)
                ])
                
                reward += alpha * avg_improvement  # Reward average improvement
                reward -= beta * distances[-1]  # Penalize current distance
                
                if distances[-1] < capture_radius:
                    reward += capture_bonus
                elif distances[-1] < alpha_capture_radius:
                    reward += 500
                elif distances[-1] < beta_capture_radius:
                    reward += 200

            reward = np.clip(reward, -2000, 2000)

            if REWARD_DEBUG:
                print(f"Complex Reward: {reward}, Distance is {distances}")

            return reward

        elif BASIC_REWARD:
            # Basic reward logic
            reward = 0
            for i, pos in enumerate(self.cca_positions):
                distance = np.linalg.norm(pos - self.foxtrot_position)
                reward -= distance / 10

            if REWARD_DEBUG:
                print(f"Basic Reward is {reward}")

            return reward

        else:
            raise ValueError("No reward flag set! Set COMPLEX_REWARD or BASIC_REWARD to True.")

if __name__ == "__main__":

    def linear_schedule(initial_value: float):
        """
        Linear schedule function for learning rate.
        Decreases linearly from `initial_value` to 0.
        """
        def schedule(progress_remaining: float):
            return progress_remaining * initial_value
        return schedule
    
    
    env = PPOEnv(grid_size=grid_size, num_cca=num_cca)
    vec_env = DummyVecEnv([lambda: PPOEnv()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    policy_kwargs = dict(
        features_extractor_class=Transformer,
        features_extractor_kwargs=dict(embed_dim=72, num_heads=8, ff_hidden=72*5, num_layers=6, seq_len=6),
        net_arch = dict(pi = [128,128,64],vf=[128,256,256,64]) #use keyword (pi) for policy network architecture -> additional ffn for decoding output, (vf) for reward func
    )

    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        normalize_advantage = True,
        use_sde = False, # Essentially reducing delta of actions when rewards are very positive (breaks it while initially learning)
        #sde_sample_freq = 3,
        learning_rate=linear_schedule(initial_value= 0.00035),
        n_steps=300, # Steps per learning update
        batch_size=50,
        gamma=0.9,
        gae_lambda= 0.75,
        vf_coef = 0.8,
        clip_range=0.4, # Clips larger updates to remain within +- 60%
        #ent_coef=0.05,
        #tensorboard_log="./Patrol&Proetect_PPO/ppo_patrol_tensorboard/"
    )

    # Cirriculum Learning

    # Train the model with stationary foxtrot (maybed small gridsize too) -> make model & env a class by itself and call it via a getter
    STATIONARY_FOXTROT = True
    COMPLEX_REWARD = True
    RECTANGULAR_FOXTROT = False
    BASIC_REWARD = False
    RAND_POS = False
    FIXED_POS = True
    RAND_FIXED_CCA = True
    model.learn(total_timesteps=50_000)

    # Continutation but now CCA's are random spawn farther away
    STATIONARY_FOXTROT = True
    COMPLEX_REWARD = True
    RECTANGULAR_FOXTROT = False
    BASIC_REWARD = False
    RAND_POS = False
    FIXED_POS = True
    RAND_FIXED_CCA = False
    model.learn(total_timesteps=100_000)

    # Continue with random stationary foxtrot (maybed small gridsize too)
    STATIONARY_FOXTROT = True
    COMPLEX_REWARD = True
    RECTANGULAR_FOXTROT = False
    BASIC_REWARD = False
    RAND_POS = True
    FIXED_POS = False
    model.learn(total_timesteps=600_000)
 
    # Finally, train it to follow a movement function
    #STATIONARY_FOXTROT = False
    #BASIC_REWARD = False
    #RECTANGULAR_FOXTROT = True
    #COMPLEX_REWARD = True
    #RAND_POS = False
    #FIXED_POS = False
    #model.learn(total_timesteps=900_000)


    # Save the model
    model.save("./PPO_V2/Trained_Model")
    print("Model Saved Succesfully!")
    os.system("pmset displaysleepnow")