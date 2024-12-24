"""
PatrolTrainingPPO.py

Refactored to use PPO (Proximal Policy Optimization) via Stable-Baselines3.
Updated to compute a collective reward for multiple robots, 
each referencing its own patrol position around the central object.
"""

import gym
import numpy as np
import torch
from math import sqrt, pi, cos, sin
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from shared_utils import (
    CentralObject, AdversarialTarget, Actor,
    normalize_state, get_patrol_positions,
    GRID_SIZE, TIME_STEP
)

# -------------------
# ENVIRONMENT DEFINITION
# -------------------

class PatrolEnv(gym.Env):
    """
    A custom Gym environment for multi-robot patrol around a central object
    with adversarial targets. Uses PPO for training.
    """
    def __init__(
        self,
        num_robots=4,
        num_targets=15,
        max_speed=10,
        patrol_radius=4.0,
        max_steps=200
    ):
        super(PatrolEnv, self).__init__()

        self.num_robots = num_robots
        self.num_targets = num_targets
        self.max_speed = max_speed
        self.patrol_radius = patrol_radius
        self.max_steps = max_steps

        # RL-specific parameters
        # (robot.x, robot.y, robot.vx, robot.vy) = 4 per robot,
        # But we are controlling only a single action (for demonstration).
        # The observation is from the perspective of "robot[0]" + 
        # relative positions to others + central object + targets.
        self.state_dim = 4 + 2 + ((self.num_robots - 1) * 2) + (self.num_targets * 2)
        self.action_size = 5  # [up, down, left, right, stay]

        # Define gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        # Discrete actions: 0=up,1=down,2=left,3=right,4=stay
        self.action_space = gym.spaces.Discrete(self.action_size)

        # Environment entities
        self.central_waypoints = [(200, 200), (800, 200), (800, 800), (200, 800)]
        self.central_obj = None
        self.robots = []
        self.targets = []

        self.current_step = 0

    def _create_targets(self):
        """
        Create adversarial targets with predefined or random waypoints
        (matching your prior logic).
        """
        waypoints_targets = [
            # 1. Circular path
            [(500 + 100 * cos(i * 2 * pi / 8), 500 + 100 * sin(i * 2 * pi / 8)) for i in range(8)],
            # 2. Square path
            [(200, 200), (200, 400), (400, 400), (400, 200)],
            # 3. Zig-zag path
            [(150, 100), (200, 300), (150, 500), (200, 700), (150, 900)],
            # 4. Random walk path
            [(np.random.randint(100, 900), np.random.randint(100, 900)) for _ in range(5)],
            # 5. Diagonal line
            [(i, i) for i in range(100, 900, 200)],
            # 6. Figure 8
            [
                (
                    500 + 100 * cos(i * pi / 4),
                    500 + 50 * sin(i * pi / 4) * (1 if i % 2 == 0 else -1)
                ) for i in range(8)
            ],
            # 7. Small circle
            [(300 + 50 * cos(i * 2 * pi / 8), 300 + 50 * sin(i * 2 * pi / 8)) for i in range(8)],
            # 8. Larger square
            [(600, 600), (600, 900), (900, 900), (900, 600)],
            # 9. Vertical zig-zag
            [(700, 100), (700, 300), (700, 500), (700, 700), (700, 900)],
            # 10. Converging inward
            [(i, 1000 - i) for i in range(100, 900, 200)],
            # 11. Horizontal zig-zag
            [(100, 500), (300, 500), (500, 500), (700, 500), (900, 500)],
            # 12. Elliptical path
            [
                (
                    500 + 150 * cos(i * 2 * pi / 8),
                    500 + 100 * sin(i * 2 * pi / 8)
                ) for i in range(8)
            ],
            # 13. Figure 8 smaller
            [
                (
                    400 + 50 * cos(i * pi / 4),
                    400 + 30 * sin(i * pi / 4) * (1 if i % 2 == 0 else -1)
                ) for i in range(8)
            ],
            # 14. Static
            [(800, 800)],
            # 15. Small random walk
            [(np.random.randint(450, 550), np.random.randint(450, 550)) for _ in range(5)],
        ]

        targets = []
        for i in range(self.num_targets):
            t = AdversarialTarget(waypoints=waypoints_targets[i], max_speed=self.max_speed)
            targets.append(t)
        return targets

    def reset(self):
        """
        Reset the environment to start a new episode.
        """
        self.current_step = 0

        # Reset central object
        self.central_obj = CentralObject(
            x=self.central_waypoints[0][0],
            y=self.central_waypoints[0][1],
            max_speed=5,
            waypoints=self.central_waypoints
        )

        # Initialize robots around patrol positions
        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)
        # Note: get_patrol_positions typically returns 4 positions if you have 4 robots
        self.robots = [Actor(px, py, self.max_speed) for (px, py) in patrol_positions]

        # Create targets
        self.targets = self._create_targets()

        # Return initial observation
        return self._get_observation()

    def _get_observation(self):
        """
        Construct the observation from the perspective of robot[0].
        """
        robot = self.robots[0]
        other_robot_deltas = []
        for r in self.robots:
            if r != robot:
                other_robot_deltas.append(r.x - robot.x)
                other_robot_deltas.append(r.y - robot.y)

        target_deltas = []
        for t in self.targets:
            target_deltas.append(t.x - robot.x)
            target_deltas.append(t.y - robot.y)

        raw_state = [
            robot.x, robot.y, robot.vx, robot.vy,
            self.central_obj.x, self.central_obj.y
        ] + other_robot_deltas + target_deltas

        obs = normalize_state(raw_state, self.state_dim, grid_size=GRID_SIZE, max_speed=self.max_speed)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Execute one time step within the environment:
        1. Move the central object
        2. Move each target
        3. Apply *the same* action to all robots (for demonstration)
        4. Compute reward
        5. Check if done
        """
        self.current_step += 1

        # Update environment entities
        self.central_obj.update()
        for t in self.targets:
            t.update()

        # If you'd like each robot to have a different action,
        # you'd need a multi-discrete or multi-agent approach
        ax, ay = self._action_to_velocity(action)

        # Control each robot with the same chosen action:
        for i, robot in enumerate(self.robots):
            robot.set_velocity(ax, ay)
            robot.update_position()

        # Compute reward, check done
        reward = self._compute_reward()  
        done = (self.current_step >= self.max_steps)
        obs = self._get_observation()
        info = {}

        return obs, reward, done, info

    def _action_to_velocity(self, action):
        """
        Convert discrete action (0..4) to velocity.
        0=up, 1=down, 2=left, 3=right, 4=stay
        """
        if action == 0:
            return (0, self.max_speed)
        elif action == 1:
            return (0, -self.max_speed)
        elif action == 2:
            return (-self.max_speed, 0)
        elif action == 3:
            return (self.max_speed, 0)
        else:
            return (0, 0)

    def _compute_reward(self):
        """
        Compute a combined reward for *all* robots.

        Each robot has its own "patrol position" from get_patrol_positions().
        We sum up partial rewards for each robot's distance to its patrol position.
        We also penalize collisions among all robots.
        """
        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)
        
        total_reward = 0.0
        # For each robot, compute distance to its assigned patrol position
        for i, robot in enumerate(self.robots):
            desired_x, desired_y = patrol_positions[i]
            rx, ry = robot.x, robot.y
            dist = sqrt((rx - desired_x)**2 + (ry - desired_y)**2)
            # Negative reward based on distance
            partial_reward = -dist / 100.0

            # Collision penalty with other robots
            for j, other_robot in enumerate(self.robots):
                if j != i:
                    other_dist = sqrt((other_robot.x - rx)**2 + (other_robot.y - ry)**2)
                    if other_dist < 0.5:
                        partial_reward -= 1.0

            total_reward += partial_reward

        # Clip total reward
        total_reward = float(np.clip(total_reward, -10, 10))
        return total_reward


# -------------------
# MAIN TRAINING LOGIC
# -------------------

def main():
    # Create environment
    env = PatrolEnv(
        num_robots=4,
        num_targets=15,
        max_speed=10,
        patrol_radius=4.0,   # Adjust as you like
        max_steps=200
    )

    # Wrap in a VecEnv for Stable-Baselines3
    vec_env = DummyVecEnv([lambda: env])

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=0.0005,
        n_steps=2048,
        batch_size=128,
        gamma=0.9,
        clip_range=0.2,
        tensorboard_log="./ppo_patrol_tensorboard/"
    )

    # Train the model
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("ppo_patrol_model")
    print("PPO model saved successfully!")


if __name__ == "__main__":
    main()