import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

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

        # Observation space: Positions of all CCA objects and Foxtrot
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(3 * (num_cca + 1),), dtype=np.float32
        )

        # Action space: Softmax for movement in 6 directions (X+, X-, Y+, Y-, Z+, Z-)
        self.action_space = spaces.Discrete(6)
        # TODO: Eventually this needs to actually be a multi discrete action space where each robot will have a different action

        # Initialize positions
        self.cca_positions = [np.array([100, 100, 100]) for _ in range(num_cca)]
        self.foxtrot_position = np.array([200, 200, 200])

    def reset(self):
        """Reset the environment to the initial state."""
        self.cca_positions = [np.array([100, 100, 100]) for _ in range(self.num_cca)]
        self.foxtrot_position = np.array([200, 200, 200])
        return self._get_observation()

    def step(self, actions):
        """Take a step in the environment."""
        # Ensure actions is iterable
        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions]

        # Update CCA positions based on actions
        for i, action in enumerate(actions):
            action = int(action)  # Convert action to integer
            self.cca_positions[i] += self._decode_action(action)

            # Clamp the position within bounds
            self.cca_positions[i] = np.clip(self.cca_positions[i], 0, self.grid_size - 1)
            if POSITIONAL_DEBUG:
                print(f"CCA {i}'s Position: {self.cca_positions[i]}")

        # Calculate reward
        reward = self._calculate_reward()

        # Update Foxtrot position with random movement
        self.foxtrot_position = foxtrot_movement_fn(self.foxtrot_position)
        if POSITIONAL_DEBUG:
            print(f"Foxtrot position: {self.foxtrot_position}")

        # Clamp Foxtrot position within bounds
        self.foxtrot_position = np.clip(self.foxtrot_position, 0, self.grid_size - 1)

        # Check if any CCA overlaps Foxtrot
        done = any(np.array_equal(pos, self.foxtrot_position) for pos in self.cca_positions)

        return self._get_observation(), reward, done, {}

    def _decode_action(self, action):
        """Decode the action into a movement vector."""
        step_size = 10
        directions = {
            0: np.array([step_size, 0, 0]),  # X+
            1: np.array([-step_size, 0, 0]), # X-
            2: np.array([0, step_size, 0]),  # Y+
            3: np.array([0, -step_size, 0]), # Y-
            4: np.array([0, 0, step_size]),  # Z+
            5: np.array([0, 0, -step_size])  # Z-
        }
        return directions[action]

    def _get_observation(self):
        """Return the current state as a flat array."""
        state = []
        for pos in self.cca_positions:
            state.extend(pos)
        state.extend(self.foxtrot_position)
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self):
        """Calculate the reward based on the distance to Foxtrot."""
        reward = 0
        for pos in self.cca_positions:
            distance = np.linalg.norm(pos - self.foxtrot_position)
            reward -= distance  # Minimize distance
            if np.array_equal(pos, self.foxtrot_position):  # Penalize overlap
                reward -= 100
        # Add a small penalty for moving out of bounds
        reward -= sum(np.any(pos < 0) or np.any(pos >= self.grid_size) for pos in self.cca_positions)
        
        if REWARD_DEBUG:
            print(f"Reward is {reward}")

        return reward


if __name__ == "__main__":
    env = PPOEnv(grid_size=grid_size, num_cca=num_cca)
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100_000)

    # Test the model
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step([action])
        if done:
            print("Episode finished")
            break

    model.save("./PPO_V2/Trained_Model")