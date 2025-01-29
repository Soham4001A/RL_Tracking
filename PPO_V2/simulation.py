import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from math import pow
from scipy.ndimage import gaussian_filter
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
    """Custom PPO Environment for controlling CCA objects."""
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

        # Terrain Utils
        self.terrain_map = None  # Will be generated if ENABLE_TERRAIN
        self.block_size = 50  # used by _generate_terrain
        self.terrain_collision_counter = 0
        self.collision_radius = 1

        # Create terrain only if enabled
        if globals.ENABLE_TERRAIN:
            self._generate_terrain()
        else:
            self.x_blocks = 0
            self.y_blocks = 0

        # ---------------------------
        # ACTION SPACE
        # ---------------------------
        self.action_space = spaces.Box(
            low=-globals.step_size, high=globals.step_size, shape=(self.num_cca, 3), dtype=np.float32
        )

        # ---------------------------
        # OBSERVATION SPACE
        # ---------------------------
        obs_shape = (
            6 * (3 * self.num_cca + 3 + 3 * self.num_cca)  # Positions, actions
            + 6 * 5 * 5 * 3  # Local terrain map history
            + 5 * 5 * 3  # Current terrain map (5x5x3)
            + 3 * self.num_cca  # Current CCA positions
        )
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(obs_shape,), dtype=np.float32
        )

        # Initialize Foxtrot positions
        if globals.RAND_POS:
            self.foxtrot_position = np.random.randint(200, 301, size=3)
        elif globals.FIXED_POS:
            self.foxtrot_position = np.array([250,250,250])

        # Initialize CCA positions (randomized in the reset function)
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
        
        self.action_history = [np.zeros((6, 3), dtype=np.float32) for _ in range(self.num_cca)]  # 6 time steps of actions per CCA
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))  # 6 rows for 5 past + current position
        self.terrain_history = [np.zeros((6, 5, 5, 3), dtype=np.float32) for _ in range(self.num_cca)] # Terrain History

        
    def reset(self, seed=None, **kwargs):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)  # Properly seed the environment

        self.cube_state = {}
        self.current_step = 0
        self.sub_step_count = 0  # Reset sub-step count too

        self.action_history = [np.zeros((6, 3), dtype=np.float32) for _ in range(self.num_cca)]  # 6 time steps of actions per CCA
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))  # 6 rows for 5 past + current position
        self.terrain_history = [np.zeros((6, 5, 5, 3), dtype=np.float32) for _ in range(self.num_cca)]

        # Regenerate terrain if enabled
        if globals.ENABLE_TERRAIN:
            self._generate_terrain()

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

        def _Foxtrot_position():
            """Update Foxtrot's position considering terrain."""
            if globals.RECTANGULAR_FOXTROT:
                # Randomly select an edge and a point along the edge
                random_edge_index = np.random.choice(len(edges))
                edge_start, edge_end = edges[random_edge_index]
                random_progress = np.random.uniform(0, 1)  # Random progress along the edge
                new_position = (1 - random_progress) * cube_vertices[edge_start] + random_progress * cube_vertices[edge_end]
                new_position = np.round(new_position).astype(int)

                # Check terrain at the new position
                terrain_z = self._terrain_z_at(int(new_position[0]), int(new_position[1]))
                if new_position[2] < terrain_z:
                    # Adjust z to avoid terrain
                    new_position[2] = terrain_z + 1  # Move 1 unit above the terrain

                self.foxtrot_position = np.clip(new_position, 0, self.grid_size - 1)

            elif globals.STATIONARY_FOXTROT:
                if globals.RAND_POS:
                    self.foxtrot_position = np.random.randint(200, 301, size=3)
                elif globals.FIXED_POS:
                    self.foxtrot_position = np.array([250, 250, 250])

            if POSITIONAL_DEBUG:
                print(f"Updated Foxtrot Position: {self.foxtrot_position}")

        _Foxtrot_position()

        # Ensure we spawn above terrain
        if globals.ENABLE_TERRAIN:
            while True:
                tz = self._terrain_z_at(int(self.foxtrot_position[0]), 
                                        int(self.foxtrot_position[1]))
                if self.foxtrot_position[2] >= tz:
                    break
                else:
                    _Foxtrot_position()

        def _CCA_position():
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

        _CCA_position()

        # Keep re-rolling below terrain if needed
        if globals.ENABLE_TERRAIN:
            for i in range(self.num_cca):
                while True:
                    tz = self._terrain_z_at(
                        int(self.cca_positions[i][0]),
                        int(self.cca_positions[i][1])
                    )
                    if self.cca_positions[i][2] >= tz:
                        break
                    else:
                        _CCA_position()

        if POSITIONAL_DEBUG:
            print(f"Foxtrot spawned at {self.foxtrot_position}")
            for i, pos in enumerate(self.cca_positions):
                print(f"CCA {i} spawned at {pos}")
                
        obs = self._get_observation()
        return obs, {}

    def step(self, actions):
        """Perform multi-action sampling for GPRO, scaling rewards without altering the initial state."""

        # Convert actions to (num_cca, 3) if needed
        actions = np.array(actions, dtype=np.float32)
        if actions.ndim == 1 and actions.size == self.num_cca * 3:
            actions = actions.reshape(self.num_cca, 3)
        elif actions.ndim == 2:
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")

        for i, action in enumerate(actions):
            self.action_history[i] = np.roll(self.action_history[i], shift=-1, axis=0)
            self.action_history[i][-1] = action
            movement_vector = self._decode_action(action)
            self.cca_positions[i] += movement_vector.astype(int)
            self.cca_positions[i] = np.clip(self.cca_positions[i], 0, self.grid_size - 1)
            
            self.terrain_history[i] = np.roll(self.terrain_history[i], shift=-1, axis=0)
            self.terrain_history[i][-1] = self._get_local_terrain_matrix(
                int(self.cca_positions[i][0]), int(self.cca_positions[i][1])
            )

        # Calculate Reward
        reward = self._calculate_reward()

        if globals.RECTANGULAR_FOXTROT:
            self.foxtrot_position = foxtrot_movement_fn_cube(self.foxtrot_position, self.cube_state, self._terrain_z_at)
            self.foxtrot_position = np.clip(self.foxtrot_position, 0, self.grid_size - 1)

        self.foxtrot_history = np.roll(self.foxtrot_history, shift=-1, axis=0)
        self.foxtrot_history[-1] = self.foxtrot_position

        self.cca_history[i] = np.roll(self.cca_history[i], shift=-1, axis=0)
        self.cca_history[i][-1] = self.cca_positions[i]


        done = False
        truncated = False

        self.current_step += 1
        if self.current_step == self.max_steps:
            done = truncated = True

        # Return the standard Gymnasium tuple
        obs = self._get_observation()
        info = {}
        return obs, reward, done, truncated, info

    # ---------------------------
    # TERRAIN GENERATION
    # ---------------------------
    def _generate_terrain(self):
        """
        Create a terrain map with smoother blocks using Gaussian filtering.
        """
        # Calculate the number of blocks in each dimension
        num_blocks_x = self.grid_size // self.block_size
        num_blocks_y = self.grid_size // self.block_size

        # Generate random z_terrain values for each block
        block_z_values = np.random.randint(low=0, high=241, size=(num_blocks_x, num_blocks_y))

        # Expand block_z_values to the full grid size
        z_values = np.repeat(np.repeat(block_z_values, self.block_size, axis=0), self.block_size, axis=1)

        # Smooth the terrain with a Gaussian filter
        z_values = gaussian_filter(z_values, sigma=self.block_size / 2)

        # Generate x and y coordinates
        x_coords, y_coords = np.meshgrid(
            np.arange(self.grid_size), np.arange(self.grid_size)
        )

        # Combine x, y, and z values into the terrain map
        self.terrain_map = np.stack((x_coords, y_coords, z_values), axis=-1)  # Shape (grid_size, grid_size, 3)

    # ---------------------------
    # LOOKUP z TERRAIN
    # ---------------------------
    def _terrain_z_at(self, x, y):
        """
        For an (x, y), find the z value directly from the high-resolution terrain_map.
        If terrain is disabled, returns 0.
        """
        if not globals.ENABLE_TERRAIN or self.terrain_map is None:
            return 0

        # Ensure x, y are within bounds
        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)

        # Direct lookup from the high-resolution terrain_map
        return self.terrain_map[x, y, 2]  # z_terrain is the third dimension

    # ---------------------------
    # LOCAL TERRAIN MATRIX
    # ---------------------------
    def _get_local_terrain_matrix(self, x, y):
        """
        Gather a 5×5 grid of points around (x, y).
        Returns a 5×5×3 matrix where each cell contains (x, y, z_terrain).
        """
        grid = np.zeros((5, 5, 3), dtype=np.float32)  # 5x5 grid with (x, y, z)

        half_range = 2  # 5×5 grid => offset range of [-2, -1, 0, 1, 2]
        for i in range(-half_range, half_range + 1):
            for j in range(-half_range, half_range + 1):
                x_val = x + i
                y_val = y + j
                z_val = 999.0  # Default value for out-of-bounds

                if 0 <= x_val < self.grid_size and 0 <= y_val < self.grid_size:
                    z_val = self._terrain_z_at(x_val, y_val)

                # Map to 5×5 grid indices
                grid[i + half_range, j + half_range] = [x_val, y_val, z_val]

        return grid  # No flattening

    def _get_observation(self):
        state = []
        for t in range(6):  # For each time step
            for i in range(self.num_cca):  # For each CCA
                state.extend(self.cca_history[i][t])  # CCA position
                state.extend(self.action_history[i][t])  # CCA action
                state.extend(self.terrain_history[i][t].flatten())  # Local terrain map history

            state.extend(self.foxtrot_history[t])  # Foxtrot position

        # Add current terrain map and CCA positions
        for i in range(self.num_cca):
            current_terrain = self._get_local_terrain_matrix(
                int(self.cca_positions[i][0]), int(self.cca_positions[i][1])
            )
            state.extend(current_terrain.flatten())  # Current terrain map
            state.extend(self.cca_positions[i])  # Current CCA position

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