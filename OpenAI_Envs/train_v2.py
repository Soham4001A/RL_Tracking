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

config_names = ["Baseline", "MHA", "LMA", "MHA_Lite"]

total_timesteps = 1_000_000  # Increased from 300k to 1M
eval_episodes = 100         # Number of episodes for evaluation
results = {}

def get_env_hyperparams(env_id):
    if env_id == "BipedalWalker-v3":
        return {
            "total_timesteps": 2_000_000,
            "learning_rate": 1e-4,
            "buffer_size": 1_000_000,
            "batch_size": 1024,
            "learning_starts": 25000,
            "ent_coef": "auto",
            "target_entropy": -4,
            "policy_kwargs": {
                "net_arch": dict(pi=[400, 300], qf=[400*2, 300*2])
            }
        }
    elif env_id == "MountainCarContinuous-v0":
        return {
            "total_timesteps": 500_000,
            "learning_rate": 7e-5,
            "buffer_size": 100_000,
            "batch_size": 512,
            "learning_starts": 5000,
            "ent_coef": "auto",
            "target_entropy": -1,
            "policy_kwargs": {
                "net_arch": dict(pi=[400, 300], qf=[400*2, 300*2])
            }
        }
    elif env_id == "Pendulum-v1":  # Pendulum-v1
        return {
            "total_timesteps": 1_000_000,
            "learning_rate": 1e-4,
            "buffer_size": 200_000,
            "batch_size": 512,
            "learning_starts": 10000,
            "ent_coef": "auto",
            "target_entropy": -2,
            "policy_kwargs": {
                "net_arch": dict(pi=[400, 300], qf=[400*2, 300*2])
            }
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

                # Dynamically set extractor_kwargs based on env observation space
                ENV_DIM = env.observation_space.shape[0]
                EMBED = max(32, ENV_DIM * 4)
                HEADS = 2 if ENV_DIM < 8 else 4
                LAYERS = 2 if ENV_DIM < 8 else 4
                extractor_kwargs = dict(
                    embed_dim=EMBED,
                    num_heads=HEADS,
                    ff_hidden=EMBED * 4,
                    num_layers=LAYERS,
                    dropout=0.05,
                    seq_len=ENV_DIM,
                )
                # ...existing code for config == 'MHA' or 'MHA_Lite'...
                if config == "MHA":
                    features_extractor_class = MHAFeaturesExtractor
                    # extractor_kwargs already set above
                elif config == "LMA":
                    features_extractor_class = LMAFeaturesExtractor
                    target_seq = ENV_DIM // 2
                    extractor_kwargs = dict(
                        embed_dim=EMBED,
                        num_heads_stacking=HEADS,
                        target_l_new=target_seq,
                        d_new=EMBED // 2,
                        num_heads_latent=HEADS,
                        ff_latent_hidden=EMBED * 2,
                        num_lma_layers=LAYERS,
                        dropout=0.05,
                        bias=True,
                        seq_len=ENV_DIM,
                    )
                elif config == "MHA_Lite":
                    features_extractor_class = MHAFeaturesExtractor
                    # extractor_kwargs already set above
                    
            # Increase number of parallel environments for better GPU utilization
            # Set rollout and gradient step parameters
            n_envs = 16
            train_freq = (1, 'step')
            gradient_steps = 16
            batch_size = 256
            
            env = make_vec_env(env_id, n_envs=n_envs)
            
            # Only apply custom feature extractor settings when not using Baseline
            if config != "Baseline":
                policy_kwargs = dict(
                    features_extractor_class=features_extractor_class,
                    features_extractor_kwargs=extractor_kwargs,
                    net_arch=dict(pi=[256, 256], qf=[512, 512])
                )
            else:
                policy_kwargs = dict(
                    net_arch=dict(pi=[256, 256], qf=[512, 512])
                )
            
            obs_dim = env.observation_space.shape[0]
            print(f"Environment Observation Dimension: {obs_dim}")
            print(f"Using Custom Feature Extractor: {config}")
                
            # Get environment-specific hyperparameters
            hyperparams = get_env_hyperparams(env_id)
            hyperparams.pop("target_entropy", None)
            
            # Update hyperparams for rollout/gradient steps and batch size
            hyperparams["train_freq"] = train_freq
            hyperparams["gradient_steps"] = gradient_steps
            hyperparams["batch_size"] = batch_size
            
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

            # Verify policy_kwargs before model creation
            with open("debug.log", "a") as f:
                f.write(f"\nPolicy kwargs for {env_id} with {config}:\n{policy_kwargs}\n")

            # Handle batched observations for feature extractors
            if features_extractor_class:
                dummy_batch = env.observation_space.sample()
                if len(dummy_batch.shape) == 1:  # Single observation
                    dummy_batch = np.stack([dummy_batch] * env.num_envs)
                dummy_batch_tensor = th.as_tensor(dummy_batch).float()
                try:
                    extractor = features_extractor_class(
                        observation_space=env.observation_space,
                        **extractor_kwargs
                    )
                    features = extractor(dummy_batch_tensor)
                    if features is None or features.shape[-1] == 0:
                        raise ValueError(f"Feature extractor returned invalid output shape: {features.shape if features is not None else None}")
                    with open("debug.log", "a") as f:
                        f.write(f"\nFeature extractor validation successful. Output shape: {features.shape}\n")
                except Exception as e:
                    with open("debug.log", "a") as f:
                        f.write(f"\nFeature extractor validation failed: {str(e)}\n")
                    raise e
                with open("debug.log", "a") as f:
                    f.write(f"\nTesting feature extractor with batched input:\n")
                    f.write(f"Input shape: {dummy_batch_tensor.shape}\n")
                features = extractor(dummy_batch_tensor)
                debug_tensor(features, "Batched feature extractor output")

            # Test feature extractor with sample input
            if features_extractor_class:
                with open("debug.log", "a") as f:
                    f.write(f"\nInput dimensions for {env_id}:\n")
                    f.write(f"Observation space shape: {env.observation_space.shape}\n")
                    f.write(f"Config seq_len: {config_seq_len}\n")
                
                # Create test batch
                obs = env.observation_space.sample()
                obs_batch = np.stack([obs] * 2)  # Create a mini-batch of size 2
                obs_tensor = th.as_tensor(obs_batch).float()
                
                # Test feature extractor
                extractor = features_extractor_class(
                    observation_space=env.observation_space,
                    **extractor_kwargs
                )
                with open("debug.log", "a") as f:
                    f.write(f"Testing with batch input shape: {obs_tensor.shape}\n")
                features = extractor(obs_tensor)
                with open("debug.log", "a") as f:
                    f.write(f"Feature extractor output shape: {features.shape}\n")

            # Merge policy_kwargs from hyperparams and custom extractor, avoiding duplicate net_arch
            merged_policy_kwargs = hyperparams.get("policy_kwargs", {}).copy()
            if config != "Baseline":
                merged_policy_kwargs["features_extractor_class"] = features_extractor_class
                merged_policy_kwargs["features_extractor_kwargs"] = extractor_kwargs

            # Create SAC model
            try:
                model = SAC(
                    "MlpPolicy",
                    env,
                    learning_rate=hyperparams["learning_rate"],
                    buffer_size=hyperparams["buffer_size"],
                    batch_size=hyperparams["batch_size"],
                    learning_starts=hyperparams["learning_starts"],
                    ent_coef=hyperparams.get("ent_coef", "auto"),
                    target_entropy=hyperparams.get("target_entropy", None),
                    train_freq=hyperparams["train_freq"],
                    gradient_steps=hyperparams["gradient_steps"],
                    policy_kwargs=merged_policy_kwargs,
                    tensorboard_log="./TensorBoardLogs",
                    verbose=1
                )
                with open("debug.log", "a") as f:
                    f.write(f"\nModel created successfully for {env_id} with {config}\n")
            except Exception as e:
                with open("debug.log", "a") as f:
                    f.write(f"\nError creating model: {str(e)}\n")
                raise e
            
            print(f"Starting training for {env_id} with {config}...")
            model.learn(total_timesteps=hyperparams["total_timesteps"], 
                       progress_bar=True, gradient_steps = 64, 
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