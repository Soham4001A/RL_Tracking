import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from classes import *

config = input("Which Model (MHA/LMA/MHA_Lite)? ")
if config!= "MHA" and config != "LMA" and config != "MHA_Lite":
    raise ValueError("Invalid Model choice. Please enter 'MHA' or 'LMA' or 'MHA_Lite.")

SEQ_LEN = 4 # Treat CartPole state as sequence of length 1
TB_LOG_DIR = "./tb_logs_cartpole_lma/"

if config == "MHA":
    # --- MHA Hyperparameters ---
    kwargs = dict(
        embed_dim=128, num_heads=4,
        ff_hidden=128*4, num_layers=4,
        seq_len=SEQ_LEN, dropout=0.1
    )

elif config == "LMA":
    kwargs = dict(
        embed_dim=128, num_heads_stacking=4, target_l_new=2, d_new=64,
        num_heads_latent=4, ff_latent_hidden=64*4, num_lma_layers=4,
        seq_len=SEQ_LEN, dropout=0.1, bias=True
    )

elif config == "MHA_Lite":
    # --- MHA Hyperparameters ---
    kwargs = dict(
        embed_dim=64, num_heads=4,
        ff_hidden=64*4, num_layers=4,
        seq_len=2, dropout=0.1
    )

# --- PPO Hyperparameters ---
ppo_kwargs = dict(
    n_steps=32,          # Increase from 128 for more stable gradients
    batch_size=256,
    n_epochs=20,           # Increase from 5 for better convergence
    gamma=0.98,           # Increase from 0.9 for better long-term credit assignment
    gae_lambda=0.8,      # Increase from 0.85 for better advantage estimation
    clip_range=0.2,
    ent_coef=0.0001,        # Add small entropy coefficient (from 0.0)
    vf_coef=0.85,          # Decrease from 0.7 to prevent value overemphasis
    max_grad_norm=0.5,
    verbose=1,
    #target_kl=0.015       # Add target KL divergence
)

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "CartPole-v1"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    vec_env = make_vec_env(
        env_id,                   # Pass environment ID string
        n_envs=num_cpu,
        seed=0,
        #monitor_dir=MONITOR_LOG_DIR, # <<< Pass monitor directory here
        # monitor_kwargs=dict(allow_early_resets=True), # Optional Monitor kwargs
        vec_env_cls=DummyVecEnv   # Or SubprocVecEnv
    )

    if config == "MHA" or config == "MHA_Lite":
        # --- Define MHA Extractor ---
        FeatureExtractor = MHAFeaturesExtractor
    elif config == "LMA":
        # --- Define LMA Extractor ---
        FeatureExtractor = LMAFeaturesExtractor
        
    # --- Define Policy Kwargs using LMA Extractor ---
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=kwargs,
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
    )

    # --- Initialize the Reward Logger Callback
    reward_logger = RewardLoggerCallback()

    # --- Initialize PPO Model ---
    print("Initializing PPO model with LMAFeaturesExtractor...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=TB_LOG_DIR, # Use separate dir for TensorBoard
        normalize_advantage=False,
        **ppo_kwargs
    )
    
    try:
        from torchinfo import summary
        summary(model.policy, input_size=(num_cpu, *vec_env.observation_space.shape))
    except ImportError: print("torchinfo not found, skipping model summary.")
    except Exception as e: print(f"Error getting model summary: {e}")
    
    train = input("Do you want to train the model? (y/n): ").strip().lower()
    if train != 'y':
        print("Skipping training.")
        vec_env.close()
        exit(0)
        
    total_timesteps = 150_000
    if config == "LMA" or config == "MHA_Lite":
        initial_lr = 1e-3  # Your initial learning rate
        final_lr = 9e-4    # Your target final learning rate
    elif config == "MHA":
        initial_lr = 2e-3
        final_lr = 1e-4
    
    lr_scheduler = LearningRateScheduler(
        initial_lr=initial_lr,
        final_lr=final_lr,
        total_timesteps=total_timesteps
    )
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[reward_logger, lr_scheduler]
    )
    
    print("Training complete.")
    
    # --- Plot the smoothed training results ---
    plt.figure(figsize=(10, 5))
    
    # Calculate rolling average with window size of 50
    window_size = 50
    smoothed_rewards = np.convolve(
        reward_logger.episode_rewards, 
        np.ones(window_size)/window_size, 
        mode='valid'
    )
    
    # Plot both raw (light) and smoothed (dark) curves
    plt.plot(reward_logger.episode_rewards, alpha=0.2, color='blue', label='Raw')
    plt.plot(smoothed_rewards, color='blue', label='Smoothed (50-episode window)')
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{config} Training Reward for CartPole-V1 over Episodes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # After training complete, add validation reward calculation
    print("\nCalculating validation reward...")
    n_eval_episodes = 100
    eval_rewards = []
    
    for _ in range(n_eval_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            episode_reward += rewards[0]
            done = dones[0]
        eval_rewards.append(episode_reward)
    
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"Average Validation Reward: {avg_reward:.2f} ± {std_reward:.2f}")
