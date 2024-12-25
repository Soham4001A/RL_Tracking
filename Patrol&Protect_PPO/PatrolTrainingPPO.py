"""
PatrolTrainingPPO.py

A single-policy, multi-discrete approach:
- We have one PPO agent controlling all robots.
- The action space is MultiDiscrete([5]*num_robots),
  i.e., each robot gets an action from {0,1,2,3,4}.
- We sum partial rewards from each robot to form the total reward.
"""

DEBUGGING = True

import gym
import numpy as np
from typing import Dict, Tuple
from math import sqrt, pi, cos, sin
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from gym.spaces import MultiDiscrete, Box

from shared_utils import (
    CentralObject, AdversarialTarget, Actor,
    normalize_state, get_patrol_positions,
    GRID_SIZE, TIME_STEP, TransformerFeatureExtractor
)

class PatrolEnv(gym.Env):
    """
    Single-policy environment with a multi-discrete action space:
    - Each of num_robots picks a discrete action from [0..4].
    - The environment sums partial rewards from each robot
      (distance to patrol positions, collisions).
    - Observations are from one "global" perspective, or you can
      just pick robot[0]'s perspective plus relative info—up to you.

    Since we want each robot to do something different but keep a single policy,
    we define:
      action_space = MultiDiscrete([5]*num_robots)
    So at each step we get an action array, e.g. [0,3,4,1].
    """

    def __init__(
        self,
        num_robots=4,
        num_targets=15,
        max_speed=10,
        patrol_radius=4.0,
        max_steps=20000
    ):
        super(PatrolEnv, self).__init__()
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.max_speed = max_speed
        self.patrol_radius = patrol_radius
        self.max_steps = max_steps

        # Define multi-discrete action space: each robot picks from 5 discrete actions
        # [up=0, down=1, left=2, right=3, stay=4]
        self.action_space = MultiDiscrete([5]*self.num_robots)

        # Observation space:
        # We'll store each robot's (x,y,vx,vy) => 4 * num_robots,
        # plus central_obj.x, central_obj.y => 2,
        # plus each target.x, target.y => 2 * num_targets.
        # total state_dim = (4*num_robots) + 2 + (2*num_targets).
        self.state_dim = (4*self.num_robots) + 2 + (2*self.num_targets)
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Entities
        self.central_waypoints = [(200, 200), (800, 200), (800, 800), (200, 800)]
        self.central_obj = None
        self.robots = []
        self.targets = []
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        # Reset central object
        self.central_obj = CentralObject(
            x=self.central_waypoints[0][0],
            y=self.central_waypoints[0][1],
            max_speed=5,
            waypoints=self.central_waypoints
        )

        # Position each robot around the central object
        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)
        # If num_robots>4, cycle or replicate
        self.robots = []
        for i in range(self.num_robots):
            px, py = patrol_positions[i % len(patrol_positions)]
            r = Actor(px, py, self.max_speed)
            self.robots.append(r)

        # Create targets
        self.targets = self._create_targets()

        return self._get_observation()

    def _create_targets(self):
        waypoints_targets = [
            [(500 + 100*cos(i*2*pi/8), 500 + 100*sin(i*2*pi/8)) for i in range(8)],
            [(200, 200), (200, 400), (400, 400), (400, 200)],
            [(150, 100), (200, 300), (150, 500), (200, 700), (150, 900)],
            [(np.random.randint(100,900), np.random.randint(100,900)) for _ in range(5)],
            [(i, i) for i in range(100,900,200)],
            [
                (
                    500 + 100*cos(i*pi/4),
                    500 + 50*sin(i*pi/4)*(1 if i%2==0 else -1)
                ) for i in range(8)
            ],
            [(300 + 50*cos(i*2*pi/8), 300 + 50*sin(i*2*pi/8)) for i in range(8)],
            [(600,600), (600,900), (900,900), (900,600)],
            [(700,100), (700,300), (700,500), (700,700), (700,900)],
            [(i, 1000 - i) for i in range(100,900,200)],
            [(100,500), (300,500), (500,500), (700,500), (900,500)],
            [
                (
                    500 + 150*cos(i*2*pi/8),
                    500 + 100*sin(i*2*pi/8)
                ) for i in range(8)
            ],
            [
                (
                    400 + 50*cos(i*pi/4),
                    400 + 30*sin(i*pi/4)*(1 if i%2==0 else -1)
                ) for i in range(8)
            ],
            [(800,800)],
            [(np.random.randint(450,550), np.random.randint(450,550)) for _ in range(5)]
        ]
        targets = []
        for i in range(self.num_targets):
            t = AdversarialTarget(waypoints=waypoints_targets[i], max_speed=self.max_speed)
            targets.append(t)
        return targets

    def step(self, action: Dict[int, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
            
        #print("Actions received inside step():", action)

        self.current_step += 1

        # Move central object, targets
        self.central_obj.update()
        for t in self.targets:
            t.update()

        # For each robot, pick velocity based on action[i]
        for i, a in enumerate(action):
            vx, vy = self._action_to_velocity(a)
            self.robots[i].set_velocity(vx, vy)

        # Update robot positions
        for r in self.robots:
            r.update_position()

        # Compute reward & check done
        reward = self._compute_reward()
        obs = self._get_observation()
        done = (self.current_step >= self.max_steps)
        info = {}
        return obs, reward, done, info

    def _action_to_velocity(self, a):
        if a == 0:
            velocity = (0, self.max_speed)  # Up
        elif a == 1:
            velocity = (0, -self.max_speed)  # Down
        elif a == 2:
            velocity = (-self.max_speed, 0)  # Left
        elif a == 3:
            velocity = (self.max_speed, 0)  # Right
        else:
            velocity = (0, 0)  # Stay
        #print(f"Action={a}, Velocity={velocity}")
        return velocity

    def _compute_reward(self):
        """
        Sum partial rewards for each robot: distance to assigned patrol position,
        plus collision penalties.
        """
        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)

        total_reward = 0.0
        # For each robot, get assigned patrol position
        for i, robot in enumerate(self.robots):
            desired_x, desired_y = patrol_positions[i % len(patrol_positions)]
            dist = sqrt((robot.x - desired_x)**2 + (robot.y - desired_y)**2)
            # e.g. negative distance penalty
            #partial = -dist / 100.0
            partial = - dist /10
            #if DEBUGGING:
            #    print(f"Distance Diff: {dist}")

            patrol_proxmity_dist = 3*sqrt(2)

            if dist < patrol_proxmity_dist:
                partial += 100
                
            # Collision penalty
            for j, other_r in enumerate(self.robots):
                if j != i:
                    d_coll = sqrt((robot.x - other_r.x)**2 + (robot.y - other_r.y)**2)
                    if d_coll < 0.5:
                        partial -= 1.0

            total_reward += partial

        if DEBUGGING:
            print(f"Total Reward: {total_reward}")

        # Clip reward
        return float(np.clip(total_reward, -1000, 1000))

    def _get_observation(self):
        """
        Single observation vector with:
         - robots => (x, y, vx, vy) * num_robots
         - central_obj => (x, y)
         - targets => (x, y) * num_targets
        """
        state = []
        for r in self.robots:
            state += [r.x, r.y, r.vx, r.vy]

        state += [self.central_obj.x, self.central_obj.y]

        for t in self.targets:
            state += [t.x, t.y]

        obs = normalize_state(
            state,
            self.state_dim,
            grid_size=GRID_SIZE,
            max_speed=self.max_speed
        )
        return np.array(obs, dtype=np.float32)

def main():
    from stable_baselines3 import PPO

    env = PatrolEnv(
        num_robots=4,
        num_targets=15,
        max_speed=10,
        patrol_radius=4.0,
        max_steps=200000
    )
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(embed_dim=64, num_heads=3, ff_hidden=128, num_layers=2),
        net_arch=[64, 64],  # Optional feedforward layers after transformer
    )

    model = PPO(
        #policy="MlpPolicy",
        policy = ActorCriticPolicy,
        #policy_kwargs={"net_arch": [256, 256, 256, 256, 128]},  # Example architecture
        env=vec_env,
        verbose=1,
        learning_rate=0.0005,
        n_steps=10000,
        batch_size=250,
        gamma=0.9,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./Patrol&Protect_PPO/ppo_patrol_tensorboard/"
    )

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)
    model.save("./Patrol&Protect_PPO/ppo_patrol_model")
    print("PPO model saved successfully!")

if __name__ == "__main__":
    main()