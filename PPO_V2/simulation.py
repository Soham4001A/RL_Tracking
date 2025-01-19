import gym
from gym import spaces
from gym.spaces import Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Internal Module Imports
from classes import *
from globals import *

DEBUG = True
REWARD_DEBUG = True
POSITIONAL_DEBUG = False

class PPOEnv(gym.Env):
    """Custom PPO Environment for controlling CCA objects."""
    def __init__(self, grid_size=500, num_cca=1):
        super(PPOEnv, self).__init__()
        self.grid_size = grid_size
        self.num_cca = num_cca
        self.cube_state = {}

        # Action space: Continuous movement in 3D space
        self.action_space = spaces.Box(
            low=-step_size, high=step_size, shape=(self.num_cca, 3), dtype=np.float32
        )

        # Observation space: Positions of all CCA objects and Foxtrot
        # Observation space: Positions of all CCA objects and Foxtrot (including history of 5 positions)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(3 * (self.num_cca + 1) * 6,), dtype=np.float32
        )

        # Initialize positions
        # Initialize positions and history
        self.cca_positions = [np.array([100, 100, 100]) for _ in range(num_cca)]
        self.foxtrot_position = np.array([200, 200, 200])

        # Initialize history for last 5 positions
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))

    def reset(self):
        """Reset the environment to the initial state."""
        self.cca_positions = [np.array([100, 100, 100]) for _ in range(self.num_cca)]
        self.foxtrot_position = np.array([200, 200, 200])
        self.cca_history = [np.tile(pos, (6, 1)) for pos in self.cca_positions]
        self.foxtrot_history = np.tile(self.foxtrot_position, (6, 1))
        return self._get_observation()

    def step(self, actions):
        """Take a step in the environment."""

        # Ensure actions is iterable
        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions]

        # Update CCA positions and history
        for i, action in enumerate(actions[:self.num_cca]):
            movement_vector = self._decode_action(action)
            self.cca_positions[i] += movement_vector.astype(int)
            self.cca_positions[i] = np.clip(self.cca_positions[i], 0, self.grid_size - 1)

            # Update history
            self.cca_history[i] = np.roll(self.cca_history[i], shift=-1, axis=0)
            self.cca_history[i][-1] = self.cca_positions[i]

        # Update Foxtrot position and history
        self.foxtrot_position = foxtrot_movement_fn_cube(self.foxtrot_position, self.cube_state)
        self.foxtrot_position = np.clip(self.foxtrot_position, 0, self.grid_size - 1)

        self.foxtrot_history = np.roll(self.foxtrot_history, shift=-1, axis=0)
        self.foxtrot_history[-1] = self.foxtrot_position

        # Calculate reward
        reward = self._calculate_reward()

        # Check if any CCA overlaps Foxtrot
        done = any(np.array_equal(pos, self.foxtrot_position) for pos in self.cca_positions)

        return self._get_observation(), reward, done, {}

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
        """Calculate the reward based on the distance to Foxtrot with aggressive scaling."""
        reward = 0
        alpha = 50  # Reward for reducing distance
        beta = 0.5  # Penalty for being far away

        for i, pos in enumerate(self.cca_positions):
            distance = np.linalg.norm(pos - self.foxtrot_position)
            prev_distance = np.linalg.norm(self.cca_history[i][-2] - self.foxtrot_position)

            # Quadratic penalty for large distances, amplified for close ranges
            reward += 1000 / (1 + distance ** 2)

            # Reward for reducing the distance
            reward += alpha * (prev_distance - distance)

            # Penalty for staying far away
            reward -= beta * distance

            # Strong penalty for overlapping with Foxtrot
            if np.array_equal(pos, self.foxtrot_position):
                reward -= 500

        # Ensure the reward is within a reasonable range for stability
        reward = np.clip(reward, -1000, 1000)

        if REWARD_DEBUG:
            print(f"Reward is {reward}")

        return reward


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
        features_extractor_kwargs=dict(embed_dim=90, num_heads=5, ff_hidden=256, num_layers=7, seq_len=6),
        #net_arch=[128, 256, 128, 64],  # Optional feedforward layers after transformer - can be specified for output to action and (future) value estimation
    )

    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        #normalize_advantage = True,
        use_sde = False, # Essentially reducing delta of actions when rewards are very positive
        #sde_sample_freq = 3,
        learning_rate=linear_schedule(initial_value= 0.003),
        n_steps=1000, # 2x the episode length which automatically terminates
        batch_size=500,
        gamma=0.9,
        gae_lambda= 0.95,
        clip_range=0.8, #Clips larger updates to remain within +- 20%
        ent_coef=0.05,
        #tensorboard_log="./Patrol&Protect_PPO/ppo_patrol_tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=10_000)

    # Save the model
    model.save("./PPO_V2/Trained_Model")
    print("Model Saved Succesfully!")