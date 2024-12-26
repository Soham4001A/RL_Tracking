"""
PatrolTrainingPPO.py

Updated with reward shaping, partial closeness reward, and normalized positions
to help the PPO agent learn more effectively.
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
    - Each of num_robots picks from 5 discrete actions [up, down, left, right, stay].
    - Summation of partial rewards for each robot.
    - Positions normalized to [0,1] range for simpler distance scale.
    """

    def __init__(
        self,
        num_robots=4,
        num_targets=15,
        max_speed=10,
        patrol_radius=4.0,
        max_steps=2000,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995
    ):
        super(PatrolEnv, self).__init__()
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.max_speed = max_speed
        self.patrol_radius = patrol_radius
        self.max_steps = max_steps
        # Epsilon parameters for exploration
        # Discrete actions [0..4] per robot
        self.action_space = MultiDiscrete([5]*self.num_robots)

        # State = (x,y,vx,vy for each robot) + (x,y central_obj) + (x,y for each target)
        self.state_dim = (4*self.num_robots) + 2 + (2*self.num_targets)
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Some large domain, e.g. 1000x1000, but we'll normalize to [0,1]
        self.central_waypoints = [(200, 200), (800, 200), (800, 800), (200, 800)]
        self.central_obj = None
        self.robots = []
        self.targets = []

        self.current_step = 0

        # Track previous distances for reward shaping
        # We store one "previous distance" per robot
        self.prev_distances = [None]*self.num_robots

    def reset(self):
        self.current_step = 0

        self.central_obj = CentralObject(
            x=self.central_waypoints[0][0],
            y=self.central_waypoints[0][1],
            max_speed=5,
            waypoints=self.central_waypoints
        )

        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)
        self.robots = []
        for i in range(self.num_robots):
            px, py = patrol_positions[i % len(patrol_positions)]
            r = Actor(px, py, self.max_speed)
            self.robots.append(r)
            self.prev_distances[i] = None  # Reset previous distance

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

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return obs, reward, done, info

    def _action_to_velocity(self, a):
        if a == 0:   # up
            velocity = (0, self.max_speed)
        elif a == 1: # down
            velocity = (0, -self.max_speed)
        elif a == 2: # left
            velocity = (-self.max_speed, 0)
        elif a == 3: # right
            velocity = (self.max_speed, 0)
        else:
            velocity = (0, 0)
        return velocity

    def _compute_reward(self):
        """
        Reward shaping approach:
          1) Convert positions to [0..1], get distance to patrol position in [0..sqrt(2)] range
          2) -dist/200 each step
          3) If distance improved from last step => +1
          4) If distance < 5 => +100 (success)
          5) small collision penalty if overlap
        """
        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)

        total_reward = 0.0
        for i, robot in enumerate(self.robots):
            # normalized position
            rx_norm = robot.x / float(GRID_SIZE)  # from [0..1000]? or 100?
            ry_norm = robot.y / float(GRID_SIZE)

            # likewise for patrol pos
            desired_x, desired_y = patrol_positions[i % len(patrol_positions)]
            desired_x_norm = desired_x / float(GRID_SIZE)
            desired_y_norm = desired_y / float(GRID_SIZE)

            #dist = sqrt((rx_norm - desired_x_norm)**2 + (ry_norm - desired_y_norm)**2)
            dist = sqrt((robot.x-desired_x)**2 + (robot.y-desired_y)**2)

            # base negative
            partial = -dist/1000

            # check improvement from last step
            if self.prev_distances[i] is not None:
                if dist < self.prev_distances[i]:
                    partial += 1.0  # small bonus if improved
            # store current distance
            self.prev_distances[i] = dist

            # success bonus
            # in normalized scale, 5 in real scale = 5/GRID_SIZE in normalized
            # for GRID_SIZE=1000 => 5/1000=0.005
            # for smaller domain => adjust accordingly
            success_thresh = 10
            if dist < success_thresh:
                partial += 100.0

            # collision penalty
            for j, other_r in enumerate(self.robots):
                if j != i:
                    dx = (robot.x - other_r.x)
                    dy = (robot.y - other_r.y)
                    if sqrt(dx*dx + dy*dy) < 1.0: # if < 1.0 => collision
                        partial -= 2.0

            total_reward += partial

        if DEBUGGING:
            print(f"Total Reward: {total_reward}")

        return float(np.clip(total_reward, -1000, 1000))

    def _get_observation(self):
        """
        Observations are normalized to [0,1].
        """
        # Build raw state
        state = []
        for r in self.robots:
            # x,y => normalized
            rx = r.x/float(GRID_SIZE)
            ry = r.y/float(GRID_SIZE)
            # vx,vy => maybe also scaled by GRID_SIZE or a speed factor
            rvx = r.vx/float(self.max_speed)
            rvy = r.vy/float(self.max_speed)
            state += [rx, ry, rvx, rvy]

        # central object
        cx = self.central_obj.x/float(GRID_SIZE)
        cy = self.central_obj.y/float(GRID_SIZE)
        state += [cx, cy]

        # targets
        for t in self.targets:
            tx = t.x/float(GRID_SIZE)
            ty = t.y/float(GRID_SIZE)
            state += [tx, ty]

        return np.array(state, dtype=np.float32)

def main():
    from stable_baselines3 import PPO

    env = PatrolEnv(
        num_robots=4,
        num_targets=15,
        max_speed=10,
        patrol_radius=4.0,
        max_steps=2000,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995
    )
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(embed_dim=72, num_heads=3, ff_hidden=128, num_layers=4, seq_len=6),
        net_arch=[432, 64],  # Optional feedforward layers after transformer
    )

    model = PPO(
        policy=ActorCriticPolicy,
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        use_sde = True,
        sde_sample_freq = 3,
        learning_rate=0.000085,
        n_steps=1000,
        batch_size=250,
        gamma=0.9,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./Patrol&Protect_PPO/ppo_patrol_tensorboard/"
    )

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)

    print(f"Final Epsilon: {env.epsilon:.4f}")  # Log final epsilon
    model.save("./Patrol&Protect_PPO/ppo_patrol_model")
    print("PPO model saved successfully!")

if __name__ == "__main__":
    main()