"""
PatrolTrainingPPO.py

Updated with reward shaping, partial closeness reward, and normalized positions
to help the PPO agent learn more effectively.
"""

DEBUGGING = True
SINGLE_ROBOT = False #Only computes reward for 1 robot

import gym
import numpy as np
from typing import Dict, Tuple
from math import sqrt, pi, cos, sin
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy,MultiInputActorCriticPolicy
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

    def __init__(  #Default Params -> These are set in the definition
        self,
        num_robots=4,
        num_targets=1,
        max_speed=10,
        patrol_radius=4.0,
        max_steps=2000,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        reward_smoothing_window=10,
        history_length = 5
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
        self.history_length = history_length

        # State = (x,y,vx,vy for each robot) + (x,y central_obj) + (x,y for each target)
        self.state_dim = (4*self.num_robots) + 2 + (2*self.num_targets)
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim * self.history_length,),
            dtype=np.float32
        )

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_smoothing_window = reward_smoothing_window
        self.reward_buffer = []
        self.state_history = []

        # Some large domain, e.g. 1000x1000, but we'll normalize to [0,1]
        #self.central_waypoints = [(800, 200), (800, 800), (200, 800), (200, 200)]
        self.central_waypoints = [(80, 20), (80, 80), (20, 80), (20, 20)]
        self.central_obj = None
        self.robots = []
        self.targets = []

        self.current_step = 0

        # Track previous distances for reward shaping
        # We store one "previous distance" per robot
        self.prev_distances = [None]*self.num_robots

    def reset(self):
        self.current_step = 0
        self.state_history = []

        self.central_obj = CentralObject(
            x=self.central_waypoints[0][0],
            y=self.central_waypoints[0][1],
            max_speed=10,
            waypoints=self.central_waypoints
        )

        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)
        self.robots = []
        for i in range(self.num_robots):
            px, py = patrol_positions[i % len(patrol_positions)]
            #r = Actor(px, py, self.max_speed)
            r = Actor(50, 50, self.max_speed) #Testing starting away from Central Obj
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

        #Testing for reducing num_targets and giving them the same path
        waypoints_targets.clear()
        for i in range (self.num_targets):
            #waypoints_targets.append([(200, 200), (200, 400), (400, 400), (400, 200)])
            waypoints_targets.append([(20, 20), (20, 40), (40, 40), (40, 20)])

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
            self.robots[i].update_position()

        # Compute reward & check done
        obs = self._get_observation()
        reward = self._compute_reward()
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
        Reward shaping approach with normalization:
        1) Convert positions to [0..1] => distance in [0..√2] if everything's within [0..1].
        2) Penalize distance proportionally (e.g., -dist_norm).
        3) If distance improved from last step => give a small positive bonus.
        4) Collision penalty is applied in normalized space.
        5) Optionally clamp reward to [-10, 1] or any range that works best for your environment.
        """
        patrol_positions = get_patrol_positions(self.central_obj, self.patrol_radius)

        total_reward = 0.0
        distance_reward = [0] * len(self.robots)

        for i, robot in enumerate(self.robots):
            # 1) Calculate normalized distance to the central object
            desired_x, desired_y = self.central_obj.x, self.central_obj.y
            dx = (robot.x - desired_x) / float(GRID_SIZE)
            dy = (robot.y - desired_y) / float(GRID_SIZE)
            dist_norm = np.sqrt(dx**2 + dy**2)

            # 2) Base negative reward for being far from the central object
            #    e.g. -dist_norm, so at max distance ~√2, it’s about -1.41

            distance_reward[i] = -dx #Split x,y??
            distance_reward[i] = -dy

            close_proxim_x = robot.x - desired_x
            close_proxim_y = robot.y - desired_y

            if close_proxim_x and close_proxim_y < 5:
                distance_reward[i] += 50

            # 3) Check improvement from the last step
            #    Compare normalized distances if you stored them previously,
            #    otherwise store the raw distance but convert back to normalized.
            if self.prev_distances[i] is not None:
                prev_dist_norm = self.prev_distances[i] / float(GRID_SIZE)
                if dist_norm < prev_dist_norm:
                    # small bonus for improvement (using normalized distance)
                    # you could reward the difference or a fixed bonus
                    distance_reward[i] += 5
            # Store current raw distance for next step’s comparison
            self.prev_distances[i] = np.sqrt((robot.x - desired_x)**2 + (robot.y - desired_y)**2)

            # 4) Collision penalty in normalized space
            #    Suppose we consider < 0.01 in normalized terms as a collision
            for j, other_r in enumerate(self.robots):
                if j != i:
                    dx_other = (robot.x - other_r.x) / float(GRID_SIZE)
                    dy_other = (robot.y - other_r.y) / float(GRID_SIZE)
                    dist_to_other_norm = np.sqrt(dx_other**2 + dy_other**2)
                    if dist_to_other_norm < 0.1:  
                        distance_reward[i] -= 0.5  # fixed penalty or scale as you like

            # (Optional) If you only compute reward for one robot
            if SINGLE_ROBOT:
                if i == 1:  # e.g., only the second robot’s reward matters
                    total_reward += distance_reward[i]
                    if DEBUGGING:
                        print(f"Total Reward: {total_reward}")
                    return total_reward

            # 5) Clamp/clip the reward in normalized space
            #    For example, clipping to [-10, 1]:
            distance_reward[i] = max(-GRID_SIZE, min(distance_reward[i], GRID_SIZE))

        total_reward += sum(distance_reward)
        total_reward = max(-GRID_SIZE, min(total_reward, GRID_SIZE))

        if DEBUGGING:
            print(f"Total STEP Reward: {total_reward}")

        return total_reward

    def _build_state(self):
        """
        Construct the normalized current state of the environment.
        - Positions normalized to [0, 1] based on GRID_SIZE.
        - Velocities normalized to [-1, 1] based on max_speed.
        """
        # Robot states: x, y, vx, vy for each robot
        robot_states = []
        for robot in self.robots:
            robot_states.extend([
                robot.x / float(GRID_SIZE),  # Normalize positions to [0, 1]
                robot.y / float(GRID_SIZE),
                robot.vx / float(self.max_speed),  # Normalize velocities to [-1, 1]
                robot.vy / float(self.max_speed)
            ])

        # Central object state: x, y
        central_state = [
            self.central_obj.x / float(GRID_SIZE),
            self.central_obj.y / float(GRID_SIZE)
        ]

        # Target states: x, y for each target
        target_states = []
        for target in self.targets:
            target_states.extend([
                target.x / float(GRID_SIZE),
                target.y / float(GRID_SIZE)
            ])

        # Combine all states into a single array
        state = np.array(robot_states + central_state + target_states, dtype=np.float32)
        return state

    def _get_observation(self):
        # Current state
        current_state = self._build_state()

        # Add to history buffer
        self.state_history.append(current_state)
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)

        # Pad history if not enough steps yet
        while len(self.state_history) < self.history_length:
            self.state_history.insert(0, np.zeros_like(current_state))

        # Flatten history into a single observation
        observation = np.concatenate(self.state_history, axis=0)

        return observation

def main():
    from stable_baselines3 import PPO

    def linear_schedule(initial_value: float):
        """
        Linear schedule function for learning rate.
        Decreases linearly from `initial_value` to 0.
        """
        def schedule(progress_remaining: float):
            return progress_remaining * initial_value
        return schedule

    env = PatrolEnv(
        num_robots=4, 
        num_targets=1, # Testing
        max_speed=10,
        patrol_radius=1.0,
        max_steps=2000,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995
    )
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(embed_dim=90, num_heads=5, ff_hidden=256, num_layers=7, seq_len=20),
        net_arch=[128, 256, 128, 64],  # Optional feedforward layers after transformer - can be specified for output to action and (future) value estimation
    )

    model = PPO(
        policy=ActorCriticPolicy,
        #policy=MultiInputActorCriticPolicy,
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=1,
        normalize_advantage = True, #??
        use_sde = False, # Essentially reducing delta of actions when rewards are very positive
        #sde_sample_freq = 3,
        learning_rate=linear_schedule(initial_value= 0.003),
        n_steps=4000, # 2x the episode length which automatically terminates
        batch_size=500,
        gamma=0.7,
        gae_lambda= 0.95,
        clip_range=0.5, #Clips larger updates to remain within +- 50%
        ent_coef=0.05,
        #tensorboard_log="./Patrol&Protect_PPO/ppo_patrol_tensorboard/"
    )

    total_timesteps = 10_000
    #total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)

    print(f"Final Epsilon: {env.epsilon:.4f}")  # Log final epsilon
    model.save("./Patrol&Protect_PPO/ppo_patrol_model")
    print("PPO model saved successfully!")

if __name__ == "__main__":
    main()