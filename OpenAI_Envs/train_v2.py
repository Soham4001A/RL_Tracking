import os
# Disable rich progress bars for TQDM to keep output clean
os.environ["TQDM_DISABLE_RICH"] = "1"

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import warnings
from classes import *  # Import custom feature extractors and utilities
from tqdm_utils import suppress_tqdm_cleanup
from sb3_contrib.common.wrappers import TimeFeatureWrapper, VecNormalize

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 0.  numpy compat shim: Ensure compatibility with older numpy versions
# -----------------------------------------------------------------------------
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "float_"):
    np.float_ = np.float64

# -----------------------------------------------------------------------------
# 1.  Table‑specific hyper‑params (A = quick, B = full)
#     These tables define different experiment settings for quick tests (A)
#     and full-scale runs (B). Each environment has its own config.
# -----------------------------------------------------------------------------
TABLE_A = {
    "Pendulum-v1": {
        "total_steps": 300_000,
        "n_envs": 8,
        "batch": 256,  # Increased batch size for stability
        "grad_steps": 8,
        "lr": 3e-4,    # Lower learning rate for stability
        "tau": 0.005,  # Slower target net update
        "net_arch": dict(pi=[32, 32], qf=[32, 32]),
        "buffer": 100_000,
    },
    "MountainCarContinuous-v0": {
        "total_steps": 300_000,
        "n_envs": 8,
        "batch": 256,
        "grad_steps": 8,
        "lr": 3e-4,
        "tau": 0.005,
        "net_arch": dict(pi=[32, 32], qf=[32, 32]),
        "buffer": 100_000,
    },
    "BipedalWalker-v3": {
        "total_steps": 600_000,
        "n_envs": 8,
        "batch": 256,
        "grad_steps": 8,
        "lr": 3e-4,
        "tau": 0.005,
        "net_arch": dict(pi=[64, 64], qf=[128, 128]),
        "buffer": 1_000_000,
    },
}

TABLE_B = {
    env: {
        "total_steps": 2_000_000 if env != "MountainCarContinuous-v0" else 1_000_000,
        "n_envs": 64,
        "batch": 1024,  # Larger batch size for stability
        "grad_steps": 64,
        "lr": 3e-4,      # Lower learning rate for stability
        "tau": 0.005,    # Slower target net update
        "net_arch": dict(pi=[128, 128], qf=[256, 256]),
        "buffer": 1_000_000,
    }
    for env in ["Pendulum-v1", "MountainCarContinuous-v0", "BipedalWalker-v3"]
}

# -----------------------------------------------------------------------------
# 2.  Safe wrapper for any extractor
#     This class wraps a feature extractor to ensure it never outputs empty features.
# -----------------------------------------------------------------------------
class SafeFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, extractor_cls=None, **kwargs):
        super().__init__(observation_space, features_dim=1)  # temp features_dim, will be overwritten
        self.inner = extractor_cls(observation_space, **kwargs)
        self._features_dim = self.inner.features_dim

    def forward(self, obs):
        x = self.inner(obs)
        if x.numel() == 0:
            raise RuntimeError("Extractor produced 0‑dim features")
        return x

# -----------------------------------------------------------------------------
# 3.  Run one (env, extractor) experiment
#     Trains and evaluates an RL agent on a given environment and feature extractor.
# -----------------------------------------------------------------------------
def run(env_id: str, table_cfg: dict, extractor_mode: str):
    cfg = table_cfg[env_id]
    # Observation normalization: add TimeFeatureWrapper and VecNormalize
    env = make_vec_env(env_id, n_envs=cfg["n_envs"], wrapper_class=TimeFeatureWrapper)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Reward scaling: add VecNormalize for reward
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=1.0)
    obs_dim = env.observation_space.shape[0]

    # -------- feature extractor selection --------
    feat_cls, feat_kwargs = None, {}
    if extractor_mode != "Baseline":
        # Dynamically set feature extractor parameters based on obs_dim
        embed = max(32, obs_dim * 4)
        heads = 2 if obs_dim < 8 else 4
        layers = 2 if obs_dim < 8 else 4
        # NOTE: The way these are setup could be improved -> They are harcoded rules but LMA reduction does not necessarily need these hardcoded rules -> For example, FFN size should be based off d_new and not a reduction factor
        if extractor_mode == "MHA":
            feat_cls, feat_kwargs = MHAFeaturesExtractor, dict(embed_dim=embed, num_heads=heads, ff_hidden=embed*4, num_layers=layers, dropout=0.05, seq_len=obs_dim)
        elif extractor_mode == "LMA":
            feat_cls, feat_kwargs = LMAFeaturesExtractor, dict(embed_dim=embed, num_heads_stacking=heads, target_l_new=obs_dim//2, d_new=embed//2, num_heads_latent=heads, ff_latent_hidden=embed*2, num_lma_layers=layers, dropout=0.05, bias=True, seq_len=obs_dim)
        elif extractor_mode == "MHA_Lite":
            feat_cls, feat_kwargs = MHAFeaturesExtractor, dict(embed_dim=embed//2, num_heads=heads, ff_hidden=(embed//2)*4, num_layers=layers, dropout=0.05, seq_len=obs_dim)

    # Set up policy network architecture and feature extractor
    policy_kwargs = dict(net_arch=cfg["net_arch"])
    # Smaller policy stdev
    policy_kwargs.update(log_std_init=-2.0, log_std_bounds=(-5, 2))
    if feat_cls:
        policy_kwargs.update(features_extractor_class=SafeFeaturesExtractor, features_extractor_kwargs=dict(extractor_cls=feat_cls, **feat_kwargs))

    # Use stable settings only for complex extractors
    if extractor_mode == "Baseline":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # Lower LR for stability
            buffer_size=cfg["buffer"],
            batch_size=cfg["batch"],
            learning_starts=cfg["batch"] * 4,
            ent_coef="auto",
            tau=cfg["tau"],
            train_freq=(1, "step"),
            gradient_steps=cfg["grad_steps"],
            policy_kwargs=policy_kwargs,
            device="cuda" if th.cuda.is_available() else "cpu",
            verbose=1,
            max_grad_norm=0.5,  # Gradient clipping
        )
    else:
        # Conservative settings for MHA/LMA/MHA_Lite
        batch_size = 256 if table_cfg is TABLE_A else 1024
        # Remove max_grad_norm from policy_kwargs (not supported by SAC)
        if "max_grad_norm" in policy_kwargs:
            del policy_kwargs["max_grad_norm"]
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=cfg["buffer"],
            batch_size=batch_size,
            learning_starts=batch_size * 4,
            ent_coef="auto",
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=cfg["grad_steps"],
            policy_kwargs=policy_kwargs,
            device="cuda" if th.cuda.is_available() else "cpu",
            verbose=1,
            max_grad_norm=0.5,  # Gradient clipping
        )

    # Train the agent
    model.learn(total_timesteps=cfg["total_steps"], progress_bar=True)

    # ------------------- deterministic evaluation -------------------
    eval_env = gym.make(env_id)
    rets = []
    for _ in range(100):
        done, ep_ret = False, 0.0
        obs, _ = eval_env.reset()
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(act)
            ep_ret += reward
            done = term or trunc
        rets.append(ep_ret)
    mean, std = float(np.mean(rets)), float(np.std(rets))
    log_line = f"{env_id:<28} | {extractor_mode:<8} | {mean:8.2f} ± {std:6.2f}\n"
    # Append evaluation results to log file
    with open("results.log", "a") as f:
        f.write(log_line)

# -----------------------------------------------------------------------------
# 4.  entry‑point
#     Main script loop: prompts user for table, then runs all experiments.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mode = input("Select table (A=quick / B=full) [A]: ").strip().lower() or "a"
    assert mode in ("a", "b"), "Please type 'A' or 'B'"
    table = TABLE_A if mode == "a" else TABLE_B

    for env in table.keys():
        #for extractor in ["Baseline", "MHA", "LMA", "MHA_Lite"]:
        for extractor in ["MHA", "LMA", "MHA_Lite"]: # Debugging Specific Feature Extractors
            run(env, table, extractor)
