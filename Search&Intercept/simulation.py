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

        # ------------------------------
        # GRPO-related parameters
        # ------------------------------
        self.sub_step_count = 0  # tracks how many sub-steps have occurred in the current macro-step
        # ------------------------------

        self.current_step = 0
        self.max_steps = 600  # Episode Length
        self.cube_state = {}

        # Reward Function Utils
        self.capture_radius = globals.step_size
        self.collision_radius = 1
        self.alpha_capture_radius = 20
        self.beta_capture_radius = 50
        self.charlie_capture_radius = 75

        # Action space: Continuous movement in 3D space
        self.action_space = spaces.MultiDiscrete([2 * step_size + 1] * (self.num_cca * 3))

        obs_dim = 6 * self.num_cca * (self.robot_observable_radius ** 3)
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
        self.sub_step_count = 0  # Reset sub-step count too

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
        for t in range(6):  # for each time step in history
            for i in range(self.num_cca):  # for each robot
                obs_list.append(self.robot_obs_history[i][t].flatten())
        return np.concatenate(obs_list).astype(np.float32)

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
        Advanced reward function optimized for exploration when target is not visible
        and focused pursuit when target is detected.
        
        Key components:
        1. Target Pursuit: Strong scaling rewards when target is visible
        2. Exploration: Incentivizes visiting low-count cells and penalizes revisiting
        3. Coordination: Rewards distribution of robots across unexplored space
        4. Memory: Maintains and rewards progress toward last known target location
        5. Movement efficiency: Penalizes excessive or redundant movement
        
        Returns combined, normalized reward across all robots.
        """
        # Hyperparameters - tuned for optimal exploration vs exploitation balance
        capture_reward = 5000.0       # High reward for reaching target
        target_visible_weight = 1000.0 # Substantial bonus just for having target visible
        distance_weight = 800.0       # Reward component based on proximity when visible
        exploration_weight = 400.0    # Base reward for exploring low-count cells
        novelty_bonus = 300.0         # Additional reward for finding new/low-count areas
        last_seen_weight = 200.0      # Weight for moving toward last known position
        coordination_weight = 150.0   # Reward for robots exploring different areas
        alignment_weight = 100.0      # Reward for moving toward target when visible
        entropy_bonus = 50.0          # Reward for diverse movement patterns
        energy_penalty = 0.1          # Small penalty for large movements
        revisit_penalty = 2.0         # Penalty for revisiting high-count cells
        time_penalty = 1.0            # Small constant penalty to encourage efficiency
        
        total_reward = 0.0
        
        # Track if any robot can see the target
        target_visible_to_any = False
        last_known_positions = []
        visitation_stats = []
        
        # First pass to gather information
        for i, robot_pos in enumerate(self.cca_positions):
            # Get current robot's subgrid and its statistics
            current_subgrid = self._extract_subgrid(robot_pos)
            current_avg = np.mean(current_subgrid)
            current_min = np.min(current_subgrid)
            current_max = np.max(current_subgrid)
            
            # Previous subgrid for comparison
            previous_subgrid = self.robot_obs_history[i][-2] if len(self.robot_obs_history[i]) >= 2 else current_subgrid
            previous_avg = np.mean(previous_subgrid)
            
            # Store stats for later coordination calculations
            visitation_stats.append((current_avg, current_min, current_max))
            
            # Check if target is visible to this robot
            half = self.robot_observable_radius // 2
            is_target_visible = np.all(self.foxtrot_position >= (robot_pos - half)) and np.all(self.foxtrot_position <= (robot_pos + half))
            
            if is_target_visible:
                target_visible_to_any = True
                last_known_positions.append((self.foxtrot_position.copy(), self.current_step))
        
        # If we have no last known positions but have class storage, initialize it
        if not hasattr(self, 'last_known_target_pos'):
            self.last_known_target_pos = None
            self.last_seen_step = -1000  # Start with a high negative value
        
        # Update last known position if target is visible
        if target_visible_to_any and last_known_positions:
            # Take the most recent sighting
            self.last_known_target_pos, self.last_seen_step = max(last_known_positions, key=lambda x: x[1])
        
        # Second pass to calculate rewards for each robot
        for i, robot_pos in enumerate(self.cca_positions):
            robot_reward = 0.0
            
            # Extract data about current and previous positions/observations
            current_subgrid = self._extract_subgrid(robot_pos)
            current_avg = visitation_stats[i][0]
            previous_subgrid = self.robot_obs_history[i][-2] if len(self.robot_obs_history[i]) >= 2 else current_subgrid
            previous_avg = np.mean(previous_subgrid)
            
            # Get robot movement
            movement = self.cca_history[i][-1] - self.cca_history[i][-2]
            movement_magnitude = np.linalg.norm(movement)
            
            # Check if target is visible to this specific robot
            half = self.robot_observable_radius // 2
            is_target_visible = np.all(self.foxtrot_position >= (robot_pos - half)) and np.all(self.foxtrot_position <= (robot_pos + half))
            
            if is_target_visible:
                # Calculate distance to target and normalized distance (0 = at target, 1 = at observation boundary)
                dist_to_target = np.linalg.norm(robot_pos - self.foxtrot_position)
                normalized_dist = np.clip(dist_to_target / self.robot_observable_radius, 0, 1)
                
                # Big reward just for having target in view
                robot_reward += target_visible_weight
                
                # Add capture reward if very close
                if dist_to_target <= self.capture_radius:
                    robot_reward += capture_reward
                
                # Add distance-based component (higher as robot gets closer)
                robot_reward += distance_weight * (1 - normalized_dist)
                
                # Add alignment reward if moving toward target
                if movement_magnitude > 0:
                    movement_unit = movement / movement_magnitude
                    target_dir = self.foxtrot_position - robot_pos
                    if np.linalg.norm(target_dir) > 0:
                        target_unit = target_dir / np.linalg.norm(target_dir)
                        alignment = np.dot(movement_unit, target_unit)
                        robot_reward += alignment_weight * np.clip(alignment, 0, 1)
            
            else:  # Target not visible - focus on exploration
                # Base exploration reward - inversely proportional to cell visit count
                # Higher reward for cells with lower visitation
                novelty_factor = np.exp(-current_avg / 2)  # Exponential decay function for novelty
                robot_reward += exploration_weight * novelty_factor
                
                # Bonus for finding area with even lower counts than before
                if current_avg < previous_avg:
                    improvement = previous_avg - current_avg
                    robot_reward += novelty_bonus * improvement
                
                # Penalty for revisiting already heavily visited cells
                if current_avg > 3:  # Only penalize if average count is significant
                    robot_reward -= revisit_penalty * current_avg
                
                # If we have a last known position and it's relatively recent, provide guidance
                if self.last_known_target_pos is not None:
                    steps_since_last_seen = self.current_step - self.last_seen_step
                    
                    # Only use last known position if it's not too old (within 100 steps)
                    if steps_since_last_seen < 100:
                        # Calculate direction to last known position
                        last_known_dir = self.last_known_target_pos - robot_pos
                        last_known_dist = np.linalg.norm(last_known_dir)
                        
                        # If robot is moving and not too close to last known pos
                        if movement_magnitude > 0 and last_known_dist > 5:
                            # Normalize directions
                            movement_unit = movement / movement_magnitude
                            last_known_unit = last_known_dir / last_known_dist
                            
                            # Calculate alignment with last known position
                            memory_alignment = np.dot(movement_unit, last_known_unit)
                            
                            # Weight decreases as time passes since last sighting
                            time_decay = max(0, 1 - steps_since_last_seen / 100)
                            robot_reward += last_seen_weight * np.clip(memory_alignment, 0, 1) * time_decay
                
                # Entropy bonus: reward for diverse movement patterns
                if len(self.action_history[i]) >= 3:
                    recent_actions = self.action_history[i][-3:]
                    # Calculate variance of recent actions as a measure of exploration diversity
                    action_variance = np.mean(np.var(recent_actions, axis=0))
                    robot_reward += entropy_bonus * min(1.0, action_variance)
            
            # Add movement efficiency penalty
            energy_cost = energy_penalty * movement_magnitude
            robot_reward -= energy_cost
            
            # Add coordination bonus if robots are exploring different areas
            # This incentivizes robots to spread out when exploring
            if len(self.cca_positions) > 1 and not is_target_visible:
                # Calculate average distance to other robots
                distances_to_others = []
                for j, other_pos in enumerate(self.cca_positions):
                    if i != j:
                        dist = np.linalg.norm(robot_pos - other_pos)
                        distances_to_others.append(dist)
                
                if distances_to_others:
                    # Higher reward for being farther from other robots (encouraging spread)
                    avg_distance = np.mean(distances_to_others)
                    normalized_distance = min(1.0, avg_distance / (self.grid_size / 4))  # Normalize by 1/4 of grid size
                    robot_reward += coordination_weight * normalized_distance
            
            # Add this robot's reward to total
            total_reward += robot_reward
        
        # Average rewards across all robots
        total_reward /= max(1, self.num_cca)
        
        # Global time penalty
        total_reward -= time_penalty
        
        if REWARD_DEBUG:
            distances_str = ", ".join([f"CCA_{i}: {np.linalg.norm(self.cca_positions[i]-self.foxtrot_position):.2f}"
                                    for i in range(self.num_cca)])
            visibility_str = ", ".join([f"CCA_{i}: {'✓' if np.all(self.foxtrot_position >= (self.cca_positions[i] - self.robot_observable_radius//2)) and np.all(self.foxtrot_position <= (self.cca_positions[i] + self.robot_observable_radius//2)) else '✗'}"
                                    for i in range(self.num_cca)])
            print(f"Step: {self.current_step} | Visibility: [{visibility_str}] | Distances: [{distances_str}] | Reward: {total_reward:.2f}")
        
        return total_reward
        
    def _extract_subgrid(self, position):
        half = self.robot_observable_radius // 2
        # Create a subgrid filled with zeros of the desired shape.
        subgrid = np.zeros((self.robot_observable_radius,
                            self.robot_observable_radius,
                            self.robot_observable_radius), dtype=np.float32)
        # Define the bounds for the subgrid in the global grid
        x_min = position[0] - half
        y_min = position[1] - half
        z_min = position[2] - half
        x_max = x_min + self.robot_observable_radius
        y_max = y_min + self.robot_observable_radius
        z_max = z_min + self.robot_observable_radius

        grid = self.grid.grid
        # Calculate the overlapping region with the global grid
        grid_x_min = max(x_min, 0)
        grid_y_min = max(y_min, 0)
        grid_z_min = max(z_min, 0)
        grid_x_max = min(x_max, self.grid_size)
        grid_y_max = min(y_max, self.grid_size)
        grid_z_max = min(z_max, self.grid_size)
        
        # Determine where to place the overlapping values in the subgrid
        sub_x_min = grid_x_min - x_min
        sub_y_min = grid_y_min - y_min
        sub_z_min = grid_z_min - z_min
        
        for xi, i in zip(range(grid_x_min, grid_x_max), range(sub_x_min, sub_x_min + (grid_x_max - grid_x_min))):
            for yj, j in zip(range(grid_y_min, grid_y_max), range(sub_y_min, sub_y_min + (grid_y_max - grid_y_min))):
                for zk, k in zip(range(grid_z_min, grid_z_max), range(sub_z_min, sub_z_min + (grid_z_max - grid_z_min))):
                    cell = grid[xi, yj, zk]
                    # If cell is a tuple, use its integer count; otherwise, use the cell value directly.
                    if isinstance(cell, tuple):
                        subgrid[i, j, k] = cell[0]
                    else:
                        subgrid[i, j, k] = cell
        return subgrid
    
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
        features_extractor_kwargs=dict(embed_dim=64, num_heads=8, ff_hidden=64*4, num_layers=4, seq_len=6),
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
        samples_per_time_step= 5,
        n_steps=600, # Steps per learning update
        batch_size=100,
        gamma=0.85,
        gae_lambda= 0.8,
        vf_coef = 0.65, # Lower reliance on v(s) to compute advantage which is then used to compute Loss -> Gradient
        clip_range=0.4, # Clips larger updates to remain within +- 60%
        #ent_coef=0.05,
        #tensorboard_log="./Patrol&Proetect_PPO/ppo_patrol_tensorboard/"
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
    model.learn(total_timesteps=30_000)

    globals.RAND_FIXED_CCA = False
    model.learn(total_timesteps=90_000)

    # Save the model
    model.save("./PPO_V2/Trained_Model")
    vec_env.save("./PPO_V2/Trained_VecNormalize.pkl")
    print("Model Saved Succesfully!")
    os.system("pmset displaysleepnow")