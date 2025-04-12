# train_compare_v2.py
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv # Needed for EvalCallback

# Import custom classes from your file
from classes import (
    MHAFeaturesExtractor,
    LMAFeaturesExtractor,
    RewardLoggerCallback,
    find_closest_divisor
)

# --- Configuration ---
ENV_IDS = ["CartPole-v1", "LunarLander-v3"]
N_EVAL_EPISODES = 100
N_ENVS = 8 # Keep 1 for simplicity unless you specifically want parallel training
TB_LOG_DIR_BASE = "./tb_logs_comparison_v2/"
SEED = 42 # Set a seed for reproducibility

# --- Environment-Specific PPO Hyperparameters (excluding feature extractor) ---
# We'll use these for BOTH baseline and custom extractor runs for a fair comparison
ENV_HYPERPARAMS = {
    "CartPole-v1": dict(
        total_timesteps=75_000,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh # Tanh often works well for CartPole
    ),
    "LunarLander-v2": dict(
        total_timesteps=250_000, # Increased steps
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01, # Small entropy bonus can help exploration
        net_arch=dict(pi=[64, 64], vf=[64, 64]), # Start with smaller net
        activation_fn=torch.nn.ReLU # ReLU is common here
    )
}

# --- Get Model Choice ---
# Added "Baseline" option
config = input("Which Model (Baseline/MHA/LMA/MHA_Lite)? ")
if config not in ["Baseline", "MHA", "LMA", "MHA_Lite"]:
    raise ValueError("Invalid Model choice. Please enter 'Baseline', 'MHA', 'LMA' or 'MHA_Lite.")

print(f"\nSelected Model Configuration: {config}")

# --- Loop through Environments ---
for env_id in ENV_IDS:
    print(f"\n===== Training on Environment: {env_id} =====")

    # --- Get Env-Specific Hyperparams ---
    if env_id not in ENV_HYPERPARAMS:
        print(f"Warning: No specific hyperparameters found for {env_id}. Using defaults.")
        # Define some defaults or fall back to CartPole settings if needed
        ppo_params = ENV_HYPERPARAMS["CartPole-v1"].copy()
    else:
        ppo_params = ENV_HYPERPARAMS[env_id].copy()

    # Extract total_timesteps as it's used outside PPO init
    total_timesteps = ppo_params.pop("total_timesteps")
    net_arch = ppo_params.pop("net_arch") # Separate net_arch for policy_kwargs if needed
    activation_fn = ppo_params.pop("activation_fn") # Separate activation_fn

    # --- Create Environment ---
    vec_env = make_vec_env(env_id, n_envs=N_ENVS, seed=SEED)

    obs_dim = vec_env.observation_space.shape[0]
    print(f"Environment Observation Dimension: {obs_dim}")

    # --- Define Feature Extractor and Policy Kwargs ---
    policy_kwargs = None
    policy_str = "MlpPolicy" # Default SB3 policy

    if config != "Baseline":
        target_seq_len = -1
        features_extractor_class = None
        extractor_kwargs = {}

        if config == "MHA":
            target_seq_len = 4
            features_extractor_class = MHAFeaturesExtractor
            # User's desired MHA extractor params
            extractor_kwargs = dict(
                embed_dim=128, num_heads=4,
                ff_hidden=128*4, num_layers=4,
                dropout=0.1
                # seq_len determined below
            )

        elif config == "LMA":
            target_seq_len = 4
            features_extractor_class = LMAFeaturesExtractor
            # User's desired LMA extractor params
            extractor_kwargs = dict(
                embed_dim=256, num_heads_stacking=8, target_l_new=2, d_new=128,
                num_heads_latent=8, ff_latent_hidden=128*4, num_lma_layers=4,
                dropout=0.1, bias=True
                 # seq_len determined below
            )

        elif config == "MHA_Lite":
            target_seq_len = 2
            features_extractor_class = MHAFeaturesExtractor
            # User's desired MHA_Lite extractor params (using MHA extractor)
            extractor_kwargs = dict(
                embed_dim=64, num_heads=4,
                ff_hidden=64*4, num_layers=4,
                dropout=0.1
                 # seq_len determined below
            )

        # --- Determine Actual seq_len based on observation dim ---
        actual_seq_len = find_closest_divisor(obs_dim, target_seq_len, max_delta=obs_dim)
        if actual_seq_len != target_seq_len:
             print(f"ADJUSTMENT ({config}): Using seq_len={actual_seq_len} for {env_id} (Obs dim: {obs_dim}) instead of target {target_seq_len}")
        extractor_kwargs['seq_len'] = actual_seq_len

        # --- Construct policy_kwargs for PPO ---
        policy_kwargs = dict(
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=extractor_kwargs,
            net_arch=net_arch, # Use the env-specific net_arch
            activation_fn=activation_fn # Use the env-specific activation
        )
        print(f"Using Custom Feature Extractor: {config}")

    else:
        # --- Baseline Configuration ---
        # For baseline, PPO uses its default MLP extractor.
        # We still need to specify the net_arch and activation_fn for the MLP.
        policy_kwargs = dict(
             net_arch=net_arch,
             activation_fn=activation_fn
        )
        print("Using Baseline (Default MLP Feature Extractor)")


    # --- Setup Logging ---
    log_name = f"{config}_{env_id}" # e.g., "LMA_LunarLander-v2"
    tb_log_path = os.path.join(TB_LOG_DIR_BASE, log_name)
    os.makedirs(tb_log_path, exist_ok=True)
    print(f"TensorBoard Log Directory: {tb_log_path}")

    reward_logger = RewardLoggerCallback() # Custom callback to store all episode rewards

    # --- Initialize and Train the Agent ---
    print(f"\nInitializing PPO model for {env_id} with {config} settings...")
    # Pass the environment-specific PPO params and the constructed policy_kwargs
    model = PPO(
        policy_str, # Should always be "MlpPolicy" when using custom extractors in SB3
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TB_LOG_DIR_BASE, # Base directory for TB logs
        device="auto",
        seed=SEED,
        **ppo_params # Unpack the environment-specific PPO hyperparameters here
    )
    print(f"Policy network structure:\n{model.policy}")


    print(f"\nStarting training for {total_timesteps} timesteps...")
    # Pass log_name to distinguish runs in TensorBoard UI
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger],
        tb_log_name=log_name, # This name groups the run within the tensorboard_log dir
        reset_num_timesteps=False
    )
    print("Training complete.")

    # --- Plot Training Rewards ---
    plt.figure(figsize=(12, 6))
    window_size = min(50, max(1, len(reward_logger.episode_rewards) // 10)) # Adaptive window
    if len(reward_logger.episode_rewards) >= window_size and window_size > 0:
        smoothed_rewards = np.convolve(
            reward_logger.episode_rewards,
            np.ones(window_size)/window_size,
            mode='valid'
        )
        smoothed_x = np.arange(window_size - 1, len(reward_logger.episode_rewards))
        plt.plot(reward_logger.episode_rewards, alpha=0.2, color='blue', label='Raw Episode Reward')
        plt.plot(smoothed_x, smoothed_rewards, color='blue', label=f'Smoothed ({window_size}-ep window)')
    elif len(reward_logger.episode_rewards) > 0:
         plt.plot(reward_logger.episode_rewards, color='blue', label='Raw Episode Reward (Few eps)')
    else:
        print("No episode rewards logged to plot.")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{config} Training Reward on {env_id} ({total_timesteps} steps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_filename = os.path.join(tb_log_path, f"{config}_{env_id}_training_rewards.png")
    plt.savefig(plot_filename)
    print(f"Saved training plot to: {plot_filename}")
    # plt.show() # Optionally disable immediate display if running many experiments

    # --- Final Evaluation ---
    print(f"\nCalculating final validation reward over {N_EVAL_EPISODES} episodes...")
    eval_rewards = []
    eval_vec_env = vec_env # Use the same vec_env for evaluation

    for i in range(N_EVAL_EPISODES):
        obs = eval_vec_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = eval_vec_env.step(action)
            episode_reward += rewards[0]
            done = dones[0]
        eval_rewards.append(episode_reward)
        # if (i+1) % 10 == 0: print(f"  Eval episode {i+1}/{N_EVAL_EPISODES} completed.")

    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"--- Environment: {env_id} ---")
    print(f"--- Model: {config} ---")
    print(f"Average Validation Reward ({N_EVAL_EPISODES} eps): {avg_reward:.2f} ± {std_reward:.2f}")
    print("----------------------------")

    # Close the environment
    vec_env.close()

print("\n===== All Environments Tested =====")
print(f"Run TensorBoard to compare results: tensorboard --logdir {TB_LOG_DIR_BASE}")