import gymnasium as gym
from gymnasium import spaces
# NO FlattenObservation here unless needed for custom wrappers NOT for default CNN
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import warnings # To warn about grid size

from classes import *

# --- Reward constants ---
FOOD_REWARD     = 5.0
DEATH_PENALTY   = -5.0
STEP_PENALTY    = -0.005 # Small penalty to encourage efficiency
CLOSER_REWARD  = 0.5 # Optional: Reward for moving closer to food
FARTHER_PENALTY = -0.1 # Optional: Penalty for moving farther from food
LOOP_PENALTY    = -0.05 # Optional: Penalize revisiting recent cells
HUNGER_PENALTY_RATE = 0.005 # Optional: Penalize not eating

class SnakeGymEnv(gym.Env):
    """
    Custom Gymnasium Environment for Snake using a CNN-compatible observation space.

    Observation Space:
        Box shape (stack_size * 4, grid_size, grid_size), uint8 [0, 255]
        Channels (concatenated across stack):
        - 0: Snake Head position (1.0 where head is, 0.0 otherwise)
        - 1: Snake Body positions (1.0 where body is, 0.0 otherwise)
        - 2: Food position (1.0 where food is, 0.0 otherwise)
        - 3: Normalized distance map to food (sqrt((x-fx)^2 + (y-fy)^2) / max_dist)

    Action Space:
        Discrete(4): 0: Up, 1: Down, 2: Left, 3: Right
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    OPPOSITES = {0:1, 1:0, 2:3, 3:2} # Action indices for opposite directions

    def __init__(self, grid_size=10, stack_size=4, seed=None):
        super().__init__()
        self.grid_size    = grid_size
        self.stack_size   = stack_size
        self.action_space = spaces.Discrete(4) # 0: Up, 1: Down, 2: Left, 3: Right

        # Observation space: (channels, height, width) as expected by SB3 CnnPolicy
        # Channels = stack_size * 4 (head, body, food, distance per frame)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.stack_size * 4, self.grid_size, self.grid_size),
            dtype=np.uint8,
        )
        # Note: SB3's default CnnPolicy expects channels-first (C, H, W).
        # If you see warnings about channels-last, ensure no wrappers are changing the order
        # and that your observation space shape is correctly defined as above.

        # Episode limits & penalties
        # Scale timeout roughly with area? Prevents trivial episodes in large grids
        self.max_steps_per_episode = 100 * grid_size # Adjusted scaling
        self.loop_penalty_steps    = 10 # How far back to check for loops if LOOP_PENALTY is used

        self.np_random = np.random.default_rng(seed)

        # Internal state variables (reset in `reset`)
        self.snake          = None # List of np.array coordinates [[y1, x1], [y2, x2], ...]
        self.food           = None # np.array coordinate [y, x]
        self.direction      = None # Current direction action index (0-3)
        self.steps          = None # Steps taken in current episode
        self.done           = None # Episode termination flag (terminated or truncated)
        self.obs_buffer     = None # List holding the last `stack_size` observations (each shape (4, H, W))
        self.last_positions = None # Optional: Store recent head positions for loop penalty
        self.steps_since_food = None # Track hunger

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) # Important for seeding in Gym v26+
        if seed is not None:
            # Re-seed the environment's RNG if a specific seed is provided
            self.np_random = np.random.default_rng(seed)

        mid = self.grid_size // 2
        # Initial snake of length 1 in the middle
        self.snake          = [np.array([mid, mid], dtype=int)]
        # Start with a random direction
        self.direction      = self.np_random.integers(0, 4)
        self.steps          = 0
        self.done           = False
        # Optional: Reset recent positions for loop penalty if used
        self.last_positions = [tuple(self.snake[0])] * self.loop_penalty_steps
        self._place_food() # Place initial food
        self.steps_since_food = 0 # Reset hunger

        # Initialize the observation buffer by duplicating the initial state
        initial_obs = self._get_obs() # Get single frame observation (4, H, W)
        self.obs_buffer = [initial_obs.copy() for _ in range(self.stack_size)]

        # Get the stacked observation (Stack*4, H, W)
        stacked_flat_channels = self._get_stacked_obs()
        info = self._get_info() # Get initial info dictionary
        return stacked_flat_channels, info

    def _place_food(self):
        """Places food randomly on an empty square."""
        while True:
            # Generate random coordinates within the grid
            pos = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            # Check if the generated position is currently occupied by the snake
            is_on_snake = any(np.array_equal(pos, seg) for seg in self.snake)
            if not is_on_snake:
                self.food = pos
                return # Food placed successfully

    def _calculate_reward(self, new_head, old_head, ate, died):
        """Calculates the reward based on game events."""
        if died:
            return DEATH_PENALTY
        if ate:
            return FOOD_REWARD

        # Basic step penalty to encourage finishing faster
        reward = STEP_PENALTY

        # --- Optional Shaping Rewards (Use with caution) ---
        # Closer/Farther shaping (Manhattan distance might be cheaper than Euclidean)
        prev_dist = np.sum(np.abs(old_head - self.food))
        new_dist = np.sum(np.abs(new_head - self.food))
        if new_dist < prev_dist:
           reward += CLOSER_REWARD
        elif new_dist > prev_dist: # Only penalize moving farther if not needed?
           reward += FARTHER_PENALTY

        # Loop Penalty (if enabled)
        if tuple(new_head) in self.last_positions[-self.loop_penalty_steps:]:
           reward += LOOP_PENALTY

        # Hunger Penalty (increases the longer the snake goes without food)
        reward -= HUNGER_PENALTY_RATE * self.steps_since_food

        return reward

    def step(self, action):
        action = int(action) # Ensure action is integer

        # Prevent snake from reversing onto itself (only possible if length > 1)
        if len(self.snake) > 1 and action == self.OPPOSITES[self.direction]:
            action = self.direction # Maintain current direction instead of reversing

        self.direction = action # Update direction

        old_head = self.snake[0].copy()
        # Define movement vectors: 0: Up (-1, 0), 1: Down (1, 0), 2: Left (0, -1), 3: Right (0, 1)
        # Note: Assuming (row, col) or (y, x) coordinates
        moves = [np.array([-1,0]), np.array([1,0]), np.array([0,-1]), np.array([0,1])]
        new_head = old_head + moves[self.direction]

        # --- Check for termination conditions ---
        # 1. Wall collision
        hit_wall = (
            new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size
        )

        # 2. Self collision (check against body segments, excluding the tail if it's about to move)
        hit_self = False
        # Only check collision if snake is long enough to potentially collide with itself
        # Check against all segments *except* the very last one if the snake didn't eat
        # (because the last segment will move away)
        check_segments = self.snake[:-1] if len(self.snake) > 1 else []
        if not hit_wall and len(check_segments) > 0: # Avoid check if already hit wall or snake is tiny
             hit_self = any(np.array_equal(new_head, seg) for seg in check_segments)

        died = hit_wall or hit_self

        # --- Check for food ---
        # Food eaten only if the snake didn't die in the same step
        ate = (not died) and np.array_equal(new_head, self.food)

        # --- Calculate Reward ---
        reward = self._calculate_reward(new_head, old_head, ate, died)

        # --- Update game state ---
        terminated = False # Standard Gymnasium term for end due to game rules (death)
        if died:
            self.done = True
            terminated = True
        else:
            # Move snake: Add new head
            self.snake.insert(0, new_head.copy())
            # Optional: Update recent positions for loop penalty
            self.last_positions.append(tuple(new_head))
            if len(self.last_positions) > self.loop_penalty_steps:
                self.last_positions.pop(0)

            if ate:
                self._place_food() # Place new food
                self.steps_since_food = 0 # Reset hunger
                # Snake grows, so we don't pop the tail
            else:
                self.snake.pop() # Remove tail segment if no food was eaten
                self.steps_since_food += 1 # Increment hunger

            self.steps += 1 # Increment step counter

        # --- Check for truncation condition (timeout) ---
        truncated = False # Standard Gymnasium term for end due to time limit
        if not self.done and self.steps >= self.max_steps_per_episode:
            truncated = True
            self.done = True # Episode ends regardless

        # --- Prepare Observation and Info ---
        current_obs = self._get_obs() # Get single frame obs (4, H, W)
        # Update observation buffer (remove oldest, add newest)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(current_obs.copy())
        # Stack frames for the final observation (Stack*4, H, W)
        stacked_flat_channels = self._get_stacked_obs()

        info = self._get_info() # Get current info dictionary
        if ate: info["food_eaten"] = True
        if died: info["termination_reason"] = "collision" # Can be refined (wall vs self)
        if truncated: info["termination_reason"] = "timeout"

        # Return conforms to Gymnasium step() output: obs, reward, terminated, truncated, info
        return stacked_flat_channels, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Generates the 4-channel observation for a single time step.
        Output shape: (4, grid_size, grid_size), dtype=float32 before scaling.
        """
        # Initialize observation grid with zeros
        o = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Snake Head
        if self.snake: # Ensure snake exists
            head_y, head_x = self.snake[0]
            # Check bounds just in case (shouldn't happen if step logic is correct)
            if 0 <= head_y < self.grid_size and 0 <= head_x < self.grid_size:
                 o[0, head_y, head_x] = 1.0

        # Channel 1: Snake Body (excluding head)
        if len(self.snake) > 1:
            for seg_y, seg_x in self.snake[1:]:
                 if 0 <= seg_y < self.grid_size and 0 <= seg_x < self.grid_size:
                    o[1, seg_y, seg_x] = 1.0

        # Channel 2: Food Location
        if self.food is not None: # Ensure food exists
             food_y, food_x = self.food
             if 0 <= food_y < self.grid_size and 0 <= food_x < self.grid_size:
                 o[2, food_y, food_x] = 1.0

             # Channel 3: Normalized Distance Map to Food
             # Create grid indices
             yy, xx = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing="ij")
             # Calculate Euclidean distance from each cell to the food
             dist = np.sqrt((yy - food_y)**2 + (xx - food_x)**2)
             # Normalize distance by the maximum possible distance in the grid (corner to corner)
             max_dist = np.sqrt(2) * (self.grid_size - 1)
             # Avoid division by zero for 1x1 grid or if max_dist is somehow zero
             o[3] = dist / max_dist if max_dist > 0 else np.zeros_like(dist)

        # Scale observation to [0, 255] and convert to uint8 for typical CNN input
        o = (o * 255).clip(0, 255).astype(np.uint8)
        return o

    def _get_stacked_obs(self):
        """
        Stacks the frames in the observation buffer along the channel dimension.
        Ensures the buffer is correctly populated, padding if necessary (e.g., at reset).
        Output shape: (stack_size * 4, grid_size, grid_size), dtype=uint8.
        """
        # Ensure buffer has stack_size elements (should be handled by reset)
        assert len(self.obs_buffer) == self.stack_size, \
            f"Observation buffer length ({len(self.obs_buffer)}) != stack size ({self.stack_size})"

        # Concatenate along the channel axis (axis 0)
        # Input list of (C, H, W) -> Output (Stack * C, H, W)
        stacked_obs = np.concatenate(self.obs_buffer, axis=0)
        return stacked_obs # Already uint8 from _get_obs

    def _get_info(self):
        """Returns auxiliary information dictionary."""
        return {
            "steps": self.steps,
            "score": len(self.snake) - 1 if self.snake else 0, # Current score (length - initial length)
            "snake_length": len(self.snake) if self.snake else 0,
            "head_pos": tuple(self.snake[0]) if self.snake else None,
            "food_pos": tuple(self.food) if self.food is not None else None,
            "steps_since_food": self.steps_since_food
        }

    def render(self, mode="human"):
        """Renders the environment state to the console."""
        if mode != "human":
            super().render(mode=mode)
            return

        # Create a character grid representation
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        # Place food ('F')
        if self.food is not None:
           # Check bounds before placing
           if 0 <= self.food[0] < self.grid_size and 0 <= self.food[1] < self.grid_size:
               grid[self.food[0], self.food[1]] = 'F'

        # Place snake ('o' for body, direction char for head)
        if self.snake:
            # Place body segments first
            for seg in self.snake[1:]:
                 if 0 <= seg[0] < self.grid_size and 0 <= seg[1] < self.grid_size:
                     grid[seg[0], seg[1]] = 'o'
            # Place head segment last (overwrites body if snake is length 1)
            h = self.snake[0]
            if 0 <= h[0] < self.grid_size and 0 <= h[1] < self.grid_size:
                # Use direction symbols for the head
                direction_char = {0: '^', 1: 'v', 2: '<', 3: '>'}.get(self.direction, 'X') # 'X' if direction is invalid
                grid[h[0], h[1]] = direction_char

        # Print the grid and stats
        print("\n" + "="*(self.grid_size*2 + 1)) # Top border
        for row in grid:
            print(' '.join(row)) # Print rows with spaces
        print(f"Score: {len(self.snake)-1 if self.snake else 0}  Steps: {self.steps or 0} Hunger: {self.steps_since_food or 0}")
        print("="*(self.grid_size*2 + 1)) # Bottom border

    def close(self):
        """Cleans up resources (if any)."""
        # No explicit resources to close in this simple example
        pass


if __name__ == "__main__":
    BASE_SEED = 12345
    NUM_ENVS  = 4  # Start with fewer envs, especially if CPU-bound or debugging
    GRID_SIZE = 36 # Default grid size
    STACK_SIZE= 6  # Number of frames to stack

    # --- Grid Size Check for Default CnnPolicy ---
    # Address Warning 3: Check grid size if using default CnnPolicy
    if GRID_SIZE < 36:
         warnings.warn(
             f"Grid size is {GRID_SIZE}x{GRID_SIZE}, which is smaller than the recommended "
             f"minimum of 36x36 for the default Stable Baselines3 CnnPolicy. "
             f"Consider increasing GRID_SIZE to 36 or larger, or use a custom "
             f"feature extractor designed for smaller inputs (like the provided "
             f"Transformer/LMA placeholders or a custom CNN). Training might be suboptimal.",
             UserWarning
         )

    # Create the vectorized environments
    # Use make_vec_env for simplicity and SubprocVecEnv for parallelism
    vec_env = make_vec_env(
        lambda: SnakeGymEnv(grid_size=GRID_SIZE, stack_size=STACK_SIZE),
        n_envs=NUM_ENVS,
        seed=BASE_SEED,
        vec_env_cls=SubprocVecEnv # Use subprocesses for true parallelism
    )

    # Address Warning 1: Wrap with VecMonitor *after* vectorization
    # This is the correct place to add the Monitor wrapper for vectorized environments.
    vec_env = VecMonitor(vec_env)

    # Sanity check the *base* environment (before vectorization/monitoring)
    print("Checking base environment...")
    check_env(SnakeGymEnv(grid_size=GRID_SIZE, stack_size=STACK_SIZE))
    print("Base environment check passed.")

    # --- Select Policy and Feature Extractor ---
    policy_kwargs = {}
    policy = "CnnPolicy" # Default policy for image-like observations
    # Note on Warning 2: The observation space defined in SnakeGymEnv is channels-first
    # (Stack*C, H, W), which is the format expected by SB3's CnnPolicy.
    # If you see warnings about channel order, ensure no intermediate wrappers
    # are changing it.

    # --- Optional: Select Custom Feature Extractor ---
    # Uncomment or modify this section if you have custom extractors
    choice = input("Use custom extractor? (mha / lma / default): ").strip().lower()

    if choice == "mha":
        print("INFO: Setting up MHA Transformer Extractor...")
        policy_kwargs = dict(
            features_extractor_class=Transformer, # Use your Transformer class
            features_extractor_kwargs=dict(
                embed_dim=64, num_heads=4,
                ff_hidden=128, num_layers=2,
                seq_len=STACK_SIZE, # Should match frame stack? Or spatial patches? Review implementation.
                dropout=0.1
            )
        )
        # Policy still needs to be compatible; CnnPolicy might work if the extractor
        # outputs a flat feature vector OR a spatial feature map compatible with the policy head.
        # Often requires a custom policy class alongside a custom extractor.
        policy = "MlpPolicy" # Or a custom policy if Transformer outputs flat features
        warnings.warn("Using MHA requires a compatible policy (e.g., MlpPolicy if features are flat) and a correctly implemented Transformer class.")

    elif choice == "lma":
        print("INFO: Setting up LMA Extractor...")
        policy_kwargs = dict(
            features_extractor_class=LMAFeaturesExtractor, # Use your LMA class
            features_extractor_kwargs=dict(
                # Pass necessary parameters for LMAFeaturesExtractor
                embed_dim=64, num_heads_stacking=4, target_l_new=3, d_new=32,
                num_heads_latent=4, ff_latent_hidden=64, num_lma_layers=2,
                seq_len=STACK_SIZE, dropout=0.1, bias=True
            )
        )
        policy = "MlpPolicy" # Or custom, similar to MHA
        warnings.warn("Using LMA requires a compatible policy and a correctly implemented LMAFeaturesExtractor class.")

    else: # Default CnnPolicy
        print("INFO: Using default Stable Baselines3 CnnPolicy.")
        policy = "CnnPolicy"
        policy_kwargs = {} # No special kwargs needed for default CNN


    # --- Configure DQN Algorithm ---
    def linear_schedule(initial_value: float):
        """Linear schedule decreasing from initial_value to 0."""
        def func(progress_remaining: float) -> float:
            """Progress remaining = 1.0 at start -> 0.0 at end"""
            return initial_value * progress_remaining
        return func

    # Hyperparameters (tune these based on performance)
    learning_rate = linear_schedule(1e-4) # Start with 1e-4 or 5e-4, schedule helps
    buffer_size = 100_000        # Size of the replay buffer (e.g., 100k - 500k)
    learning_starts = 10_000     # Steps to collect before training starts (e.g., 10k-50k)
    # Consider batch_size relative to buffer and learning starts
    #batch_size = 256             # Total batch size across all envs (e.g., 64, 128, 256)
    batch_size = 32 * NUM_ENVS # Alternative: Scale with NUM_ENVS (test this)
    exploration_fraction = 0.3   # Fraction of total steps for exploration decay (e.g., 0.1 to 0.5)
    exploration_final_eps = 0.05 # Final epsilon value (e.g., 0.01 to 0.1)
    target_update_interval = 1000 # Steps between updating the target network (e.g., 500 - 10000)
    train_freq = 4               # Train the model every N steps collected (across all envs)
    gradient_steps = 1           # How many gradient steps to perform per training cycle (usually 1 for DQN)
    total_timesteps = 1_000_000  # Total training steps (increase significantly for harder tasks, e.g., 1M, 5M, 10M+)

    model = DQN(
        policy,
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=1.0,                   # Tau=1.0 means hard updates for target network
        gamma=0.99,                # Discount factor
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        verbose=1,                 # Print training progress
        tensorboard_log="./snake_tensorboard/", # Directory for TensorBoard logs
        seed=BASE_SEED             # Seed the RL algorithm initialization
    )

    # --- Setup Evaluation Callback ---
    # Create a separate environment for evaluation
    eval_env = make_vec_env(
        lambda: SnakeGymEnv(grid_size=GRID_SIZE, stack_size=STACK_SIZE),
        n_envs=1, # Typically use 1 environment for deterministic evaluation
        seed=BASE_SEED + NUM_ENVS, # Use a different seed for eval env
    )
    # Wrap eval env with VecMonitor to log evaluation stats
    eval_env = VecMonitor(eval_env, filename="./logs/eval_monitor_log")

    # Callback saves the best model based on evaluation reward
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/results',
        eval_freq=max(10000 // NUM_ENVS, 500), # Evaluate every N training steps
        n_eval_episodes=10,        # Number of episodes to run for evaluation
        deterministic=True,        # Use deterministic actions for evaluation
        render=False               # Don't render during evaluation
    )

    # --- Train the Agent ---
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Using policy: {policy}")
    if policy_kwargs:
        print(f"With policy_kwargs: {policy_kwargs}")
    print(f"Hyperparameters: LR={learning_rate(1.0):.1e} schedule, Buffer={buffer_size}, Batch={batch_size}, Start={learning_starts}, ExpFraction={exploration_fraction}, FinalEps={exploration_final_eps}, TargetUpdate={target_update_interval}")

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10, # Log training stats every 10 episodes/updates
        callback=eval_callback # Use the evaluation callback
    )
    model.save("snake_dqn_final") # Save the final model state
    print("Training finished.")

    # --- Evaluate the Best Model ---
    print("\nLoading best model for final evaluation...")
    # Load the best model saved by EvalCallback
    # Note: If training was short or no improvement was found, 'best_model.zip' might not exist.
    try:
        model = DQN.load("./logs/best_model/best_model", env=eval_env)
        print("Best model loaded.")
    except FileNotFoundError:
        print("WARNING: No best model found. Loading final model instead.")
        model = DQN.load("snake_dqn_final", env=eval_env)


    # Run evaluation episodes on the monitored evaluation environment
    eval_episodes = 50
    total_rewards = []
    total_scores = []
    total_lengths = []

    print(f"Running final evaluation for {eval_episodes} episodes...")
    obs = eval_env.reset()
    episodes_done = 0
    while episodes_done < eval_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

        # Check if an episode finished (VecMonitor handles aggregation)
        if done[0]: # Check the first (and only) environment in eval_env
             # Extract episode stats from the info dict provided by VecMonitor
             episode_info = info[0].get("episode")
             if episode_info:
                 print(f"  Eval Episode {episodes_done + 1}: Reward={episode_info['r']:.2f}, Length={episode_info['l']}, Score={episode_info['l']-1}")
                 total_rewards.append(episode_info['r'])
                 total_lengths.append(episode_info['l'])
                 # Calculate score based on length (assuming snake starts at length 1)
                 total_scores.append(episode_info['l'] - 1)
                 episodes_done += 1
             # No need to manually reset, VecMonitor handles it via AutoResetWrapper

    # Calculate and print average statistics
    if total_rewards:
        avg_r = float(np.mean(total_rewards))
        avg_score = float(np.mean(total_scores))
        avg_len = float(np.mean(total_lengths))
        std_r = float(np.std(total_rewards))
        std_score = float(np.std(total_scores))
        print("-" * 30)
        print(f"Evaluation over {episodes_done} episodes:")
        print(f"  Avg Reward: {avg_r:.3f} +/- {std_r:.3f}")
        print(f"  Avg Score:  {avg_score:.3f} +/- {std_score:.3f}")
        print(f"  Avg Length: {avg_len:.3f}")
        print("-" * 30)
    else:
        print("No evaluation episodes completed.")


    # --- Visualize Trained Agent ---
    print("\nStarting visualization of the trained agent...")
    vis_env = SnakeGymEnv(grid_size=GRID_SIZE, stack_size=STACK_SIZE, seed=BASE_SEED+NUM_ENVS+1) # Use a different seed
    # No VecMonitor or make_vec_env needed for single visualization env

    for ep in range(5): # Visualize 5 episodes
        obs, _ = vis_env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_steps = 0
        print(f"\n=== Visualization Episode {ep+1} ===")
        while not (done or truncated):
            vis_env.render() # Render the current state
            # model.predict usually handles single Box observation correctly
            action, _ = model.predict(obs, deterministic=True) # Use deterministic actions for visualization
            obs, reward, done, truncated, info = vis_env.step(action)
            ep_reward += reward
            ep_steps += 1
            import time
            time.sleep(0.1) # Add delay to make visualization watchable

        vis_env.render() # Render the final state
        print(f"Episode {ep+1} finished:")
        print(f"  Reward: {ep_reward:.2f}")
        print(f"  Score: {info['score']}")
        print(f"  Steps: {ep_steps}")
        if info.get("termination_reason"):
            print(f"  Reason: {info['termination_reason']}")
        time.sleep(1) # Pause before next visualization episode

    # Clean up environments
    print("Closing environments...")
    vec_env.close()
    eval_env.close()
    vis_env.close()
    print("Done.")