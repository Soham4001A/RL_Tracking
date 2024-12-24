import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from shared_utils import CentralObject, AdversarialTarget, Actor, get_patrol_positions
from PatrolTrainingPPO import PatrolEnv

GRID_SIZE = 1000
EPISODE_STEPS = 200

def run_showcase(model_path="ppo_patrol_model.zip"):
    # 1. Create environment
    env = PatrolEnv(
        num_robots=4,
        num_targets=15,
        max_speed=10,
        patrol_radius=4.0,
        max_steps=EPISODE_STEPS
    )
    vec_env = DummyVecEnv([lambda: env])

    # 2. Load model
    model = PPO.load(model_path, vec_env)
    print("PPO model loaded successfully for showcase!")

    # 3. Reset the vectorized env
    obs = vec_env.reset()

    # 4. Access the actual environment after reset
    real_env = vec_env.envs[0]

    # Visualization setup
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_title("Patrol & Protect Showcase (PPO)")

    # Now that reset is called, real_env.central_obj should exist
    central_dot, = ax.plot([real_env.central_obj.x],
                           [real_env.central_obj.y],
                           'r*', markersize=10, label='Central Object')

    colors = ['bo', 'go', 'ro', 'mo']
    robot_dots = [ax.plot([], [], c)[0] for c in colors]
    target_dots = [ax.plot([], [], 'yo')[0] for _ in range(real_env.num_targets)]
    ax.legend()

    def init():
        for dot in robot_dots + target_dots:
            dot.set_data([], [])
        return robot_dots + target_dots + [central_dot]

    def update(frame):
        nonlocal obs
        # Use the model to get the next action
        action, _states = model.predict(obs, deterministic=True)
        next_obs, rewards, done, info = vec_env.step(action)

        # Update references
        updated_env = vec_env.envs[0]
        central_obj = updated_env.central_obj
        robots = updated_env.robots
        targets = updated_env.targets

        # Update the visualization
        central_dot.set_data([central_obj.x], [central_obj.y])
        for i, robot in enumerate(robots):
            robot_dots[i].set_data([robot.x], [robot.y])
        for i, t in enumerate(targets):
            target_dots[i].set_data([t.x], [t.y])

        obs = next_obs
        if done[0]:
            obs = vec_env.reset()

        return robot_dots + target_dots + [central_dot]

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=EPISODE_STEPS,
        interval=50,
        blit=True
    )
    plt.show()

if __name__ == "__main__":
    run_showcase(model_path="/Users/sohamsane/Documents/Coding Projects/ObjectTrackingRL/Patrol&Protect_PPO/ppo_patrol_model.zip")