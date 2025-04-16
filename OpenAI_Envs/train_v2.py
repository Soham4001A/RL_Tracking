import os
os.environ["TQDM_DISABLE_RICH"] = "1"

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import warnings
from classes import *
from tqdm_utils import suppress_tqdm_cleanup

warnings.filterwarnings("ignore")

# Patch for numpy.bool8 and numpy.float_ deprecation
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# List of environments (only continuous action spaces for SAC)
env_names = [
    "BipedalWalker-v3",           # Continuous action space
    "MountainCarContinuous-v0",   # Continuous action space
    "Pendulum-v1",                # Continuous action space
]

# --- Get Model Choice ---
# Added "Baseline" option
#config = input("Which Model (Baseline/MHA/LMA/MHA_Lite)? ")
#if config not in ["Baseline", "MHA", "LMA", "MHA_Lite"]:
#    raise ValueError("Invalid Model choice. Please enter 'Baseline', 'MHA', 'LMA' or 'MHA_Lite.")

config_names = ["MHA", "LMA", "MHA_Lite"]

total_timesteps = 1_000_000  # Increased from 300k to 1M
eval_episodes = 100         # Number of episodes for evaluation
results = {}

def get_env_hyperparams(env_id):
    if env_id == "BipedalWalker-v3":
        return {
            "total_timesteps": 2_000_000,
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_starts": 10000
        }
    elif env_id == "MountainCarContinuous-v0":
        return {
            "total_timesteps": 500_000,
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_starts": 1000
        }
    else:  # Pendulum-v1
        return {
            "total_timesteps": 1_000_000,
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_starts": 1000
        }

def debug_tensor(tensor, name):
    with open("debug.log", "a") as f:
        if isinstance(tensor, torch.Tensor):
            f.write(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}\n")
            f.write(f"{name} values: {tensor}\n")
        elif isinstance(tensor, np.ndarray):
            f.write(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}\n")
            f.write(f"{name} values: {tensor}\n")
        else:
            f.write(f"{name} type: {type(tensor)}\n")
            f.write(f"{name} value: {tensor}\n")

def train_and_evaluate(env_id, config):
    with suppress_tqdm_cleanup():
        try:
            # --- Define Feature Extractor and Policy Kwargs ---
            policy_kwargs = None
            
            if env_id == "CartPole-v1":
                config_seq_len = 4
            elif env_id == "Acrobot-v1":
                config_seq_len = 6
            elif env_id == "BipedalWalker-v3":
                config_seq_len = 24
            elif env_id == "MountainCarContinuous-v0":
                config_seq_len = 2
            elif env_id == "Pendulum-v1":
                config_seq_len = 3
            
            if config != "Baseline":
                target_seq_len = -1
                features_extractor_class = None
                extractor_kwargs = {}

                if config == "MHA":
                    features_extractor_class = MHAFeaturesExtractor
                    # User's desired MHA extractor params
                    extractor_kwargs = dict(
                        embed_dim=128, num_heads=4,
                        ff_hidden=128*4, num_layers=4,
                        dropout=0.1, seq_len = config_seq_len
                    )

                elif config == "LMA":
                    features_extractor_class = LMAFeaturesExtractor
                    target_seq = config_seq_len/2
                    if type(target_seq) == float:
                        target_seq = int(target_seq)
                    # User's desired LMA extractor params
                    extractor_kwargs = dict(
                        embed_dim=128, num_heads_stacking=4, target_l_new=target_seq, d_new=64,
                        num_heads_latent=4, ff_latent_hidden=64*4, num_lma_layers=4,
                        dropout=0.1, bias=True, seq_len = config_seq_len
                    )

                elif config == "MHA_Lite":
                    features_extractor_class = MHAFeaturesExtractor
                    # User's desired MHA_Lite extractor params (using MHA extractor)
                    extractor_kwargs = dict(
                        embed_dim=64, num_heads=4,
                        ff_hidden=64*4, num_layers=4,
                        dropout=0.1, seq_len = config_seq_len 
                    )
                    
            # Increase number of parallel environments for better GPU utilization
            env = make_vec_env(env_id, n_envs=64)  # Changed from 25 to 64
            
            # Update policy kwargs with optimized network architecture
            policy_kwargs = dict(
                features_extractor_class=features_extractor_class,
                features_extractor_kwargs=extractor_kwargs,
                net_arch=dict(pi=[256, 256], qf=[512, 512])  # Wider networks
            )
            
            obs_dim = env.observation_space.shape[0]
            print(f"Environment Observation Dimension: {obs_dim}")
            print(f"Using Custom Feature Extractor: {config}")
                
            # Get environment-specific hyperparameters
            hyperparams = get_env_hyperparams(env_id)
            
            if features_extractor_class:
                with open("debug.log", "a") as f:
                    f.write(f"\nTesting feature extractor for {env_id} with {config}\n")
                # Test feature extractor
                dummy_obs = env.observation_space.sample()
                dummy_obs_tensor = th.as_tensor(dummy_obs).float()
                extractor = features_extractor_class(
                    observation_space=env.observation_space,
                    **extractor_kwargs
                )
                features = extractor(dummy_obs_tensor)
                debug_tensor(features, "Feature extractor output")

            # Create SAC model with environment-specific hyperparameters
            model = SAC("MlpPolicy", 
                       env, 
                       learning_rate=hyperparams["learning_rate"],
                       buffer_size=hyperparams["buffer_size"],
                       batch_size=hyperparams["batch_size"],
                       tau=hyperparams["tau"],
                       gamma=hyperparams["gamma"],
                       ent_coef=hyperparams.get("ent_coef", "auto"),
                       target_entropy=hyperparams.get("target_entropy", None),
                       train_freq=hyperparams.get("train_freq", 1),
                       gradient_steps=hyperparams.get("gradient_steps", 1),
                       learning_starts=hyperparams.get("learning_starts", 10000),
                       policy_kwargs=policy_kwargs, 
                       tensorboard_log="./TensorBoardLogs", 
                       verbose=1)
            
            print(f"Starting training for {env_id} with {config}...")
            model.learn(total_timesteps=hyperparams["total_timesteps"], 
                       progress_bar=True, 
                       log_interval=100)  # Added log_interval
            print(f"Finished training {env_id} with {config}")

            # Evaluation
            eval_env = gym.make(env_id)
            episode_rewards = []
            for ep in range(eval_episodes):
                with open("results.log", "a") as f:
                    f.write(f"Starting evaluation episode {ep} for {env_id} with {config}\n")
                try:
                    obs, _ = eval_env.reset()
                    with open("debug.log", "a") as f:
                        f.write(f"\nEpisode {ep} initial observation:\n")
                    debug_tensor(obs, "Initial observation")
                    
                    done = False
                    total_reward = 0
                    while not done:
                        try:
                            # Convert observation to tensor and debug
                            obs_tensor = th.as_tensor(obs).float()
                            debug_tensor(obs_tensor, "Model input")
                            
                            action, _ = model.predict(obs, deterministic=True)
                            debug_tensor(action, "Model output action")
                            
                            obs, reward, term, trunc, _ = eval_env.step(action)
                            debug_tensor(reward, "Environment reward")
                            
                            if reward is None:
                                with open("debug.log", "a") as f:
                                    f.write("Warning: Received None reward\n")
                                continue
                                
                            total_reward += reward
                            done = bool(term or trunc)
                            
                        except Exception as e:
                            with open("debug.log", "a") as f:
                                f.write(f"Error in episode step: {str(e)}\n")
                            break
                            
                    episode_rewards.append(total_reward)
                    
                except Exception as e:
                    with open("debug.log", "a") as f:
                        f.write(f"Error in episode {ep}: {str(e)}\n")
                    
            if len(episode_rewards) == 0:
                results[env_id] = {'mean': 'Error: No episodes completed during evaluation', 'std': None}
            else:
                avg_reward = float(np.mean(episode_rewards))
                std_reward = float(np.std(episode_rewards))
                results[env_id] = {'mean': avg_reward, 'std': std_reward}
        except Exception as e:
            results[env_id] = {'mean': f"Error: {e}", 'std': None}

# Run
for env in env_names:
    for config in config_names:
        print(f"\n--- Training and Evaluating {env} with {config} ---")
        train_and_evaluate(env,config)
        # Log results for this (env, config) pair
        with open("results.log", "a") as f:
            f.write(f"Results for {env} with {config}: {results[env]}")
            f.write(f"{env} ({config}): Mean = {results[env]['mean']}, Std = {results[env]['std']}\n")