import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from math import pow
from stable_baselines3 import PPO, GPRO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# Internal Module Imports
from classes import *
import globals

DEBUG = True
REWARD_DEBUG = True
POSITIONAL_DEBUG = True

class PPOEnv(gym.Env):
    """Custom Env that supports sub-step logic for GPRO."""

    def __init__(self, grid_size=500, num_cca=1):
        super(PPOEnv, self).__init__()
        self.grid_size = grid_size
        self.num_cca = num_cca

        # ------------------------------
        # GPRO-related parameters
        # ------------------------------
        self.sub_step_count = 0  # tracks how many sub-steps have occurred in the current macro-step
        # ------------------------------

        self.current_step = 0
        self.max_steps = 400  # The "macro-step" horizon
        self.cube_state = {}

        # Reward Function Utils
        self.capture_radius = globals.step_size
        self.collision_radius = 1
        self.alpha_capture_radius = 20
        self.beta_capture_radius = 50
        self.charlie_capture_radius = 75

        # Action space: Continuous movement in 3D space
        #  shape=(self.num_cca, 3), each dimension in [-step_size, step_size]
        self.action_space = spaces.Box(
            low=-step_size, high=step_size, shape=(self.num_cca, 3), dtype=np.float32
        )

        # Observations: 6 time steps worth of (CCA pos, Foxtrot pos, CCA action)
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(6 * (3 * self.num_cca + 3 + 3 * self.num_cca),),
            dtype=np.float32,
        )

        # Initialize Foxtrot position
        if globals.RAND_POS:
            self.foxtrot_position = np.random.randint(200, 301, size=3)
        elif globals.FIXED_POS:
            self.foxtrot_position = np.array([250, 250, 250])

        # Initialize CCA positions (randomized in reset())
        if globals.RAND_FIXED_CCA:
            self.cca_positions = [
                np.clip(
                    self.foxtrot_position + np.random.randint(-spawn_range, spawn_range + 1, size=3),
                    0, self.grid_size - 1
                ) for _ in range(self.num_cca)
            ]
        elif globals.PROXIMITY_CCA:
            self.cca_positions = [self.foxtrot_position for _ in range(self.num_cca)]
        else:
            self.cca_positions = [
                np.random.randint(0, self.grid_size, size=3) for _ in range(self.num_cca)
            ]

        self.action_history = [np.zeros((6, 3), dtype=np.float32) for _ in range(self.num_cca)]
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))  # 6 rows for 5 past + current position

        
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.cube_state = {}
        # Reset counters
        self.current_step = 0
        self.sub_step_count = 0  # Reset sub-step count too

        self.action_history = [np.zeros((6, 3), dtype=np.float32) for _ in range(self.num_cca)]
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))

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

        if globals.RECTANGULAR_FOXTROT:
            # Randomly select an edge and a point along the edge
            random_edge_index = np.random.choice(len(edges))
            edge_start, edge_end = edges[random_edge_index]
            random_progress = np.random.uniform(0, 1)  # Random progress along the edge
            self.foxtrot_position = (1 - random_progress) * cube_vertices[edge_start] + random_progress * cube_vertices[edge_end]
            self.foxtrot_position = np.round(self.foxtrot_position).astype(int)

        if globals.STATIONARY_FOXTROT:
            if globals.RAND_POS:
                self.foxtrot_position = np.random.randint(200, 301, size=3)
            elif globals.FIXED_POS:
                self.foxtrot_position = np.array([250,250,250])

        # Create CCA Positions
        if globals.RAND_FIXED_CCA:
            self.cca_positions = [
                np.clip(
                    self.foxtrot_position + np.random.randint(-spawn_range, spawn_range + 1, size=3),
                    0, self.grid_size - 1
                ) for _ in range(self.num_cca)
            ]
        
        elif globals.PROXIMITY_CCA:
            self.cca_positions = [self.foxtrot_position for _ in range(self.num_cca)]

        else:
            self.cca_positions = [
            np.random.randint(0, self.grid_size, size=3) for _ in range(self.num_cca)
            ]

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
        """Perform one sub-step, accumulate reward, but only increment the major step after enough sub-steps."""

        self.current_step += 1

        # Convert actions to (num_cca, 3) if needed
        actions = np.array(actions, dtype=np.float32)
        if actions.ndim == 1 and actions.size == self.num_cca * 3:
            actions = actions.reshape(self.num_cca, 3)
        elif actions.ndim == 2:
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")

        # Update CCA positions, action histories, etc.
        for i, action in enumerate(actions):
            self.action_history[i] = np.roll(self.action_history[i], shift=-1, axis=0)
            self.action_history[i][-1] = action
            movement_vector = self._decode_action(action)
            self.cca_positions[i] += movement_vector.astype(int)
            self.cca_positions[i] = np.clip(self.cca_positions[i], 0, self.grid_size - 1)

            self.cca_history[i] = np.roll(self.cca_history[i], shift=-1, axis=0)
            self.cca_history[i][-1] = self.cca_positions[i]

        # Calculate reward for *this* sub-step
        reward = self._calculate_reward()

        # Possibly move Foxtrot if it's a moving target:
        if globals.RECTANGULAR_FOXTROT:
            self.foxtrot_position = foxtrot_movement_fn_cube(self.foxtrot_position, self.cube_state)
            self.foxtrot_position = np.clip(self.foxtrot_position, 0, self.grid_size - 1)

        self.foxtrot_history = np.roll(self.foxtrot_history, shift=-1, axis=0)
        self.foxtrot_history[-1] = self.foxtrot_position

        done = False
        truncated = False

        if self.current_step == self.max_steps:
            done = truncated = True

        # Return the standard Gymnasium tuple
        obs = self._get_observation()
        info = {}
        return obs, reward, done, truncated, info

    def _get_observation(self):
        state = []
        for t in range(6):  # For each time step
            for i in range(self.num_cca):  # For each CCA
                state.extend(self.cca_history[i][t])  # CCA position
                state.extend(self.action_history[i][t])  # CCA action
            state.extend(self.foxtrot_history[t])  # Foxtrot position
        return np.array(state, dtype=np.float32)

    def _decode_action(self, action):
        """Clamp action values to ensure they stay within bounds."""
        return np.clip(action, -step_size, step_size)

    def _calculate_reward(self):
        """
        Refactored reward function using potential-based reward shaping and multi-objective optimization.
        """
        if globals.COMPLEX_REWARD:
            # Initialize reward
            reward = 0.0

            # Hyperparameters
            alpha = 300.0              # Weight for progress
            beta = 0.05                # Weight for energy efficiency
            gamma_collision = -2000.0  # Penalty for collisions
            gamma = 0.1                # Potential shaping weight
            max_action_norm = 10.0     # Maximum expected action magnitude
            capture_radius = 15

            # Define the potential function
            def potential():
                return -np.mean([np.linalg.norm(pos - self.foxtrot_position) for pos in self.cca_positions])

            # Current and previous potential
            current_potential = potential()
            previous_potential = getattr(self, 'previous_potential', current_potential)

            # Potential-based shaping
            shaped_reward = gamma * (current_potential - previous_potential)
            reward += shaped_reward

            # Update previous potential for next step
            self.previous_potential = current_potential

            # Progress-Based Reward
            # Calculate average progress across all CCAs
            progress = 0.0
            for i, pos in enumerate(self.cca_positions):
                if len(self.cca_history[i]) >= 2:
                    prev_distance = np.linalg.norm(self.cca_history[i][-2] - self.foxtrot_position)
                    current_distance = np.linalg.norm(pos - self.foxtrot_position)
                    progress += (prev_distance - current_distance)

                    if current_distance < capture_radius: #capture radius bonus
                        reward += 1000

            progress /= self.num_cca
            reward += alpha * progress

            # Energy Efficiency Penalty
            energy_penalty = beta * np.sum([np.linalg.norm(action) for action in self.action_history[i][-1] for i in range(self.num_cca)])
            reward -= energy_penalty

            # Collision Penalty
            collision_penalty = 0.0
            for i, pos in enumerate(self.cca_positions):
                distance = np.linalg.norm(pos - self.foxtrot_position)
                if distance < self.collision_radius:
                    collision_penalty += gamma_collision
            reward += collision_penalty

            # Optional: Exploration Bonus
            # Encourage movement by rewarding the agent for covering more distance over time
            movement_bonus = 0.0
            for i, pos in enumerate(self.cca_positions):
                total_movement = np.linalg.norm(pos - self.cca_history[i][0])
                movement_bonus += 0.1 * total_movement  # Tunable parameter
            reward += movement_bonus

            # Clip reward to prevent extreme values
            reward = np.clip(reward, gamma_collision, 2000)

            if REWARD_DEBUG:
                #print(f"Refactored Complex Reward: {reward}")
                for i in range(self.num_cca):
                    recent_distance = np.linalg.norm(self.cca_positions[i] - self.foxtrot_position)
                    print(f"Raw Complex Reward: {reward}, CCA {i} Distance: {recent_distance}")

            return reward

        elif globals.BASIC_REWARD:
            # Basic reward logic remains unchanged or can be similarly refactored
            reward = 0
            for i, pos in enumerate(self.cca_positions):
                distance = np.linalg.norm(pos - self.foxtrot_position)
                reward -= distance / 100

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
    
    # Initial Ciriculum Flags need to be set before env creation
    globals.STATIONARY_FOXTROT = False
    globals.RECTANGULAR_FOXTROT = True
    globals.COMPLEX_REWARD = True 
    globals.RAND_POS = True #This is for stationary foxtrot & TODO: should be rewritten as so
    globals.FIXED_POS = False
    globals.RAND_FIXED_CCA = False
    globals.PROXIMITY_CCA = True

    env = PPOEnv(grid_size=500, num_cca=1) 
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    policy_kwargs = dict(
        features_extractor_class=Transformer,
        features_extractor_kwargs=dict(embed_dim=64, num_heads=8, ff_hidden=64*4, num_layers=4, seq_len=6),
        net_arch = dict(pi = [128,128,64],vf=[128,256,256,64]) #use keyword (pi) for policy network architecture -> additional ffn for decoding output, (vf) for reward func
    )

    model = GPRO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        normalize_advantage = True,
        use_sde = False, # Essentially reducing delta of actions when rewards are very positive (breaks it while initially learning)
        #sde_sample_freq = 3,
        learning_rate=linear_schedule(initial_value= 0.0002),
        samples_per_time_step= 5,
        n_steps=600, # Steps per learning update
        batch_size=100,
        gamma=0.85,
        gae_lambda= 0.8,
        vf_coef = 0.85, # Lower reliance on v(s) to compute advantage which is then used to compute Loss -> Gradient
        clip_range=0.4, # Clips larger updates to remain within +- 60%
        #ent_coef=0.05,
        #tensorboard_log="./Patrol&Proetect_PPO/ppo_patrol_tensorboard/"
    )

    # Cirriculum Learning
    globals.COMPLEX_REWARD = True 


    """
    # Train the model with stationary foxtrot and small random CCA
    globals.RECTANGULAR_FOXTROT = False
    globals.STATIONARY_FOXTROT = True
    globals.RAND_POS = False #This is for stationary foxtrot & TODO: should be rewritten as so
    globals.FIXED_POS = True #This is for stationary foxtrot & TODO: should be rewritten as so
    globals.RAND_FIXED_CCA = True
    model.learn(total_timesteps=70_000)

    # Continutation but now CCA's are random spawn farther away
    globals.RECTANGULAR_FOXTROT = False
    globals.STATIONARY_FOXTROT = True
    globals.RAND_POS = False #This is for stationary foxtrot & TODO: should be rewritten as so
    globals.FIXED_POS = True #This is for stationary foxtrot & TODO: should be rewritten as so
    globals.RAND_FIXED_CCA = False
    model.learn(total_timesteps=140_000)

    # Continue with random stationary foxtrot and random spawn CCA (maybed small gridsize too)
    globals.RECTANGULAR_FOXTROT = False
    globals.STATIONARY_FOXTROT = True
    globals.FIXED_POS = False #This is for stationary foxtrot & TODO: should be rewritten as so
    globals.RAND_POS = True #This is for stationary foxtrot & TODO: should be rewritten as so
    globals.RAND_FIXED_CCA = False
    model.learn(total_timesteps=350_000) 
    
    """
    # Finally, train it to follow a movement function - spawn CCA's at same location as foxtrot initially
    globals.STATIONARY_FOXTROT = False
    globals.RECTANGULAR_FOXTROT = True
    globals.RAND_FIXED_CCA = False
    globals.PROXIMITY_CCA = True
    model.learn(total_timesteps=90_000)

    globals.PROXIMITY_CCA = False
    model.learn(total_timesteps=90_000)

    # Save the model
    model.save("./PPO_V2/Trained_Model")
    vec_env.save("./PPO_V2/Trained_VecNormalize.pkl")
    print("Model Saved Succesfully!")
    os.system("pmset displaysleepnow")