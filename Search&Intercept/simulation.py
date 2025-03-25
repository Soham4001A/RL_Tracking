import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from math import pow
from stable_baselines3 import PPO, GRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# Internal Module Imports
from classes import *
import globals

DEBUG = True
REWARD_DEBUG = True
POSITIONAL_DEBUG = True

class PPOEnv(gym.Env):
    """Custom Env that supports sub-step logic for GRPO."""

    def __init__(self, grid_size=500, num_cca=3):
        super(PPOEnv, self).__init__()
        self.grid_size = grid_size
        self.num_cca = num_cca
        # Initialize grid using the GridSpace class 
        self.grid = GridSpace(self.grid_size)
        # Define the observable radius for a robot (adjustable as needed)
        self.robot_observable_radius = 100

        self.current_step = 0
        self.max_steps = 600  # Episode Length
        self.cube_state = {}

        # Action space: Continuous movement in 3D space
        self.action_space = spaces.MultiDiscrete([2 * step_size + 1] * (self.num_cca * 3))

        # In the __init__ method of PPOEnv, update the observation space dimension:
        obs_dim = 6 * self.num_cca * 5 + (self.num_cca * 3)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(obs_dim,),
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

        self.action_history = [np.zeros((6, 3), dtype=np.float32) for _ in range(self.num_cca)]
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))
        # After initializing self.cca_positions and self.foxtrot_history
        self.robot_obs_history = [ [self._extract_subgrid(pos) for _ in range(6)] for pos in self.cca_positions ]

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
        # Store previous positions for grid update
        prev_cca_positions = [pos.copy() for pos in self.cca_positions]
        prev_foxtrot_position = self.foxtrot_position.copy()

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

        # Update the grid for each robot's movement using the GridSpace class
        for i, new_pos in enumerate(self.cca_positions):
            self.grid.update_robot_area(prev_cca_positions[i], new_pos, f"CCA_{i}", self.robot_observable_radius)
            self.robot_obs_history[i].pop(0)
            self.robot_obs_history[i].append(self._extract_subgrid(new_pos))

        # Update the grid for the target's movement
        self.grid.update_target_area(prev_foxtrot_position, self.foxtrot_position)

        done = False
        truncated = False

        if self.current_step == self.max_steps:
            done = truncated = True

        # Return the standard Gymnasium tuple
        obs = self._get_observation()
        info = {}
        return obs, reward, done, truncated, info

    def _get_observation(self):
        obs_list = []
        for t in range(6):  # For each time step in history
            for i in range(self.num_cca):  # For each robot
                obs_list.append(self.robot_obs_history[i][t].flatten())
        # Append positions of all robots (each position is 3 values)
        robot_positions = np.concatenate([pos for pos in self.cca_positions])
        return np.concatenate(obs_list + [robot_positions]).astype(np.float32)

    def _decode_action(self, action):
        """Decode a discrete action for a single robot into a movement vector in [-step_size, step_size]."""
        action = np.array(action, dtype=np.float32)
        # If the action is a single robot's action (shape (3,)), use it directly
        if action.ndim == 1 and action.size == 3:
            return action - globals.step_size
        # If the action is for multiple robots (shape (num_cca, 3)), decode each accordingly
        elif action.ndim == 2 and action.shape[1] == 3:
            return action - globals.step_size
        else:
            raise ValueError(f"Unexpected action shape in _decode_action: {action.shape}")

    def _calculate_reward(self, obs=None, action=None):
        """
        Updated reward function using switch-case logic:
        - Case 1: No target in sight and no last known location: Encourage robots to spread out and search.
        - Case 2: Target is in sight: Heavily reward reducing distance and aligning movement toward the target.
        - Case 3: No target in sight but a last known location exists: Encourage movement toward the last known target, with incentive decaying over time.
        
        Additional penalties for energy expenditure and a constant time penalty are applied.
        """
        # Hyperparameters (tweak as necessary)
        spread_reward_weight = 200.0       # Reward for spreading out when target is unknown.
        intercept_reward_weight = 1000.0   # Reward for reducing distance to target when target is visible.
        alignment_reward_weight = 500.0    # Bonus for aligning movement toward the target.
        last_known_reward_weight = 500.0     # Reward for moving toward the last known location.
        decay_factor = 0.01                # Decay per time step since last seen.
        energy_penalty_weight = 0.1        # Penalty for large movements.
        time_penalty = 1.0                 # Constant penalty per time step.

        total_reward = 0.0
        num = self.num_cca

        # --- Compute spread reward (for Case 1) ---
        spread_reward = 0.0
        for i in range(num):
            for j in range(i+1, num):
                dist = np.linalg.norm(self.cca_positions[i] - self.cca_positions[j])
                spread_reward += dist
        if num > 1:
            spread_reward /= (num * (num-1) / 2)

        # --- Determine target visibility for each robot ---
        target_in_sight = False
        robot_visible = [False] * num
        half = self.robot_observable_radius // 2
        for i, pos in enumerate(self.cca_positions):
            if np.all(self.foxtrot_position >= (pos - half)) and np.all(self.foxtrot_position <= (pos + half)):
                robot_visible[i] = True
                target_in_sight = True

        # --- Check for last known target location ---
        last_known_available = (hasattr(self, 'last_known_target_pos') and self.last_known_target_pos is not None)
        if last_known_available:
            time_since_last = self.current_step - self.last_seen_step
        else:
            time_since_last = None

        # --- Switch-case reward logic ---
        # Case 2: Target in sight
        if target_in_sight:
            for i, pos in enumerate(self.cca_positions):
                dist = np.linalg.norm(pos - self.foxtrot_position)
                # Reward inversely proportional to distance
                reward_i = intercept_reward_weight * (1 - np.clip(dist / self.robot_observable_radius, 0, 1))
                # Extra bonus for the robot that sees the target: alignment bonus
                if robot_visible[i]:
                    movement = self.cca_history[i][-1] - self.cca_history[i][-2]
                    if np.linalg.norm(movement) > 0:
                        movement_unit = movement / np.linalg.norm(movement)
                        target_dir = self.foxtrot_position - pos
                        if np.linalg.norm(target_dir) > 0:
                            target_unit = target_dir / np.linalg.norm(target_dir)
                            alignment = np.dot(movement_unit, target_unit)
                        else:
                            alignment = 1.0
                    else:
                        alignment = 0.0
                    reward_i += alignment_reward_weight * np.clip(alignment, 0, 1)
                total_reward += reward_i
            total_reward /= num

        # Case 3: No target in sight but last known target available
        elif last_known_available:
            # Decay factor reduces incentive as time since last seen increases
            decay = max(0, 1 - decay_factor * (self.current_step - self.last_seen_step))
            for i, pos in enumerate(self.cca_positions):
                dist = np.linalg.norm(pos - self.last_known_target_pos)
                reward_i = last_known_reward_weight * decay * (1 - np.clip(dist / self.robot_observable_radius, 0, 1))
                total_reward += reward_i
            total_reward /= num

        # Case 1: No target in sight and no last known target
        else:
            # Encourage exploration by rewarding robot dispersion
            total_reward = spread_reward_weight * (spread_reward / self.grid_size)

        # --- Energy penalty: Penalize excessive movement ---
        energy_penalty_total = 0.0
        for i in range(num):
            movement = self.cca_history[i][-1] - self.cca_history[i][-2]
            energy_penalty_total += np.linalg.norm(movement)
        energy_penalty_total *= energy_penalty_weight
        total_reward -= energy_penalty_total

        # --- Global time penalty ---
        total_reward -= time_penalty

        # --- Update last known target location if target is visible ---
        if target_in_sight:
            self.last_known_target_pos = self.foxtrot_position.copy()
            self.last_seen_step = self.current_step

        if REWARD_DEBUG:
            vis_str = ", ".join([f"CCA_{i}: {'✓' if robot_visible[i] else '✗'}" for i in range(num)])
            dist_str = ", ".join([f"{np.linalg.norm(self.cca_positions[i]-self.foxtrot_position):.2f}" for i in range(num)])
            print(f"Step {self.current_step} | Visibility: [{vis_str}] | Distances: [{dist_str}] | Reward: {total_reward:.2f}")

        return total_reward
        
    # Updated _extract_subgrid() method:
    # Add the new _extract_observation() method to PPOEnv (replace the old _extract_subgrid if present):
    def _extract_subgrid(self, position):
        """
        Compute a summary observation vector for the observable region around a robot.
        The observable region is a cube of side length `self.robot_observable_radius` centered on `position`.
        If the target (Foxtrot) is within this region, return:
            [ -1, target_x, target_y, target_z, 0 ]
        Otherwise, return:
            [ sum, last_known_target_x, last_known_target_y, last_known_target_z, counter ]
        where `sum` is the sum of all grid cell values in the observable region,
        and `counter` is the number of time steps since the target was last seen (or -1 if never seen).
        """
        half = self.robot_observable_radius // 2
        # Define bounds for the observable region
        x_min = position[0] - half
        y_min = position[1] - half
        z_min = position[2] - half
        x_max = x_min + self.robot_observable_radius
        y_max = y_min + self.robot_observable_radius
        z_max = z_min + self.robot_observable_radius

        # Clip bounds to grid limits
        x_min_clip = max(x_min, 0)
        y_min_clip = max(y_min, 0)
        z_min_clip = max(z_min, 0)
        x_max_clip = min(x_max, self.grid_size)
        y_max_clip = min(y_max, self.grid_size)
        z_max_clip = min(z_max, self.grid_size)

        # Extract the subgrid using slicing (this is vectorized)
        grid_slice = self.grid.grid[x_min_clip:x_max_clip, y_min_clip:y_max_clip, z_min_clip:z_max_clip]

        # Define a vectorized function to extract the integer value of each cell
        get_value = np.vectorize(lambda cell: cell[0] if isinstance(cell, tuple) else cell)
        cell_values = get_value(grid_slice)
        sum_val = np.sum(cell_values)

        # Check if target is within the observable region
        region_min = np.array([x_min, y_min, z_min])
        region_max = np.array([x_max, y_max, z_max])
        target_visible = np.all(self.foxtrot_position >= region_min) and np.all(self.foxtrot_position < region_max)

        if target_visible:
            obs_vec = np.array([-1, self.foxtrot_position[0], self.foxtrot_position[1], self.foxtrot_position[2], 0], dtype=np.float32)
            self.last_known_target_pos = self.foxtrot_position.copy()
            self.last_seen_step = self.current_step
        else:
            if hasattr(self, 'last_known_target_pos') and self.last_known_target_pos is not None:
                last_known = self.last_known_target_pos
                counter = self.current_step - self.last_seen_step
            else:
                last_known = np.array([0, 0, 0], dtype=np.float32)
                counter = -1
            obs_vec = np.array([sum_val, last_known[0], last_known[1], last_known[2], counter], dtype=np.float32)
        return obs_vec
    
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

    env = PPOEnv(grid_size=500, num_cca=3) 
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    policy_kwargs = dict(
        features_extractor_class=Transformer,
        features_extractor_kwargs=dict(embed_dim=64, num_heads=4, ff_hidden=64*3, num_layers=3, seq_len=3),
        net_arch = dict(pi = [128,128,64],vf=[128,256,256,64]) #use keyword (pi) for policy network architecture -> additional ffn for decoding output, (vf) for reward func
    )

    model = GRPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        reward_function=env._calculate_reward, 
        verbose=1,
        normalize_advantage = True,
        use_sde = False, # Essentially reducing delta of actions when rewards are very positive (breaks it while initially learning)
        #sde_sample_freq = 3,
        learning_rate=linear_schedule(initial_value= 0.0003),
        samples_per_time_step= 2,
        n_steps=600, # Steps per learning update
        batch_size=100,
        gamma=0.85,
        gae_lambda= 0.8,
        vf_coef = 0.65, # Lower reliance on v(s) to compute advantage which is then used to compute Loss -> Gradient
        clip_range=0.4, # Clips larger updates to remain within +- 60%
        #ent_coef=0.05,
        #tensorboard_log="./Search&Intercept/logs/"
    )

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
    globals.STATIONARY_FOXTROT = True
    globals.RECTANGULAR_FOXTROT = False
    globals.RAND_FIXED_CCA = True
    globals.PROXIMITY_CCA = False
    model.learn(total_timesteps=180_000)

    globals.RAND_FIXED_CCA = False
    model.learn(total_timesteps=180_000)

    # Save the model
    model.save("./PPO_V2/Trained_Model")
    vec_env.save("./PPO_V2/Trained_VecNormalize.pkl")
    print("Model Saved Succesfully!")
    os.system("pmset displaysleepnow")