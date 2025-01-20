import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from math import pow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Internal Module Imports
from classes import *
from globals import (
    grid_size,
    num_cca,
    step_size,
    STATIONARY_FOXTROT,
    RECTANGULAR_FOXTROT,
    COMPLEX_REWARD, 
    BASIC_REWARD
)

DEBUG = True
REWARD_DEBUG = True
POSITIONAL_DEBUG = False

class PPOEnv(gym.Env):
    """Custom PPO Environment for controlling CCA objects."""
    def __init__(self, grid_size=500, num_cca=1):
        super(PPOEnv, self).__init__()
        self.grid_size = grid_size
        self.num_cca = num_cca
        self.current_step = 0
        self.max_steps = 10_000 # Epsiode Length in the simulation
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

        # Initialize positions (randomized in the reset function)
        self.cca_positions = [np.array([100, 100, 100]) for _ in range(num_cca)]
        self.foxtrot_position = np.random.randint(0, self.grid_size, size=3)

        # Initialize history for last 5 positions
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))  # 6 rows for 5 past + current position

    def reset(self, seed=None, **kwargs):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)  # Properly seed the environment

        self.cube_state = {}
        self.current_step = 0
        
        # Randomize CCA positions and reset history
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
            self.foxtrot_position = np.random.randint(0, self.grid_size, size=3)

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
        if STATIONARY_FOXTROT:
            self.foxtrot_position = np.random.randint(0, self.grid_size, size=3)
            
        elif RECTANGULAR_FOXTROT:
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
        """Return the current state as a flat array."""
        state = []
        for i in range(self.num_cca):
            state.extend(self.cca_history[i].flatten())  # Add CCA history
        state.extend(self.foxtrot_history.flatten())  # Add Foxtrot history
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
            beta = 0.2
            capture_bonus = 1000.0
            capture_radius = 5.0
            alpha_capture_radius = 50
            beta_capture_radius = 150

            for i, pos in enumerate(self.cca_positions):
                distance = np.linalg.norm(pos - self.foxtrot_position)
                prev_distance = np.linalg.norm(self.cca_history[i][-2] - self.foxtrot_position)
                improvement = (prev_distance - distance) / prev_distance if prev_distance > 0 else 0
                reward += alpha * improvement
                reward -= beta * distance
                if distance < capture_radius:
                    reward += capture_bonus
                if distance < alpha_capture_radius:
                    reward += 500
                if distance < beta_capture_radius:
                    reward += 30

            reward = np.clip(reward, -5000, 5000)

            if REWARD_DEBUG:
                print(f"Complex Reward: {reward}")

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
    vec_env = DummyVecEnv([lambda: env])
    policy_kwargs = dict(
        features_extractor_class=Transformer,
        features_extractor_kwargs=dict(embed_dim=90, num_heads=6, ff_hidden=256, num_layers=8, seq_len=6),
        net_arch = dict(vf=[128,256,256,64]) #use keyword (pi) for policy network architecture -> additional ffn for decoding output, (vf) for reward func
    )

    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        normalize_advantage = True,
        use_sde = False, # Essentially reducing delta of actions when rewards are very positive
        #sde_sample_freq = 3,
        learning_rate=linear_schedule(initial_value= 0.0002),
        n_steps=3000, # Steps per learning update
        batch_size=300,
        gamma=0.9,
        gae_lambda= 0.35,
        vf_coef = 0.8,
        clip_range=0.8, # Clips larger updates to remain within +- 20%
        #ent_coef=0.05,
        #tensorboard_log="./Patrol&Proetect_PPO/ppo_patrol_tensorboard/"
    )

    # Cirriculum Learning

    # Train the model with stationary foxtrot (maybe add small gridsize too)
    STATIONARY_FOXTROT = True
    COMPLEX_REWARD = True
    RECTANGULAR_FOXTROT = False
    BASIC_REWARD = False
    model.learn(total_timesteps=100_000)
 
    #STATIONARY_FOXTROT = False
    #BASIC_REWARD = False
    #RECTANGULAR_FOXTROT = True
    #COMPLEX_REWARD = True
    # Train the model with moving foxtrot
    #model.learn(total_timesteps=300_000)


    # Save the model
    model.save("./PPO_V2/Trained_Model")
    print("Model Saved Succesfully!")