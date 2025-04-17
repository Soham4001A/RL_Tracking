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
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.vec_env import VecNormalize
from torch.optim.lr_scheduler import LinearLR

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

class LRSchedulerCallback(BaseCallback):
    def __init__(self, scheduler_factory, verbose=0):
        super().__init__(verbose)
        self.scheduler_factory = scheduler_factory
        self.schedulers = []
        self.attached = False
    def _on_step(self) -> bool:
        if not self.attached:
            # Attach schedulers to all optimizers (policy and critics)
            optimizers = []
            policy_opt = getattr(self.model.policy, 'optimizer', None)
            if policy_opt is not None:
                optimizers.append(policy_opt)
            # Try to get critic optimizers (SB3: self.model.critic.optimizer or self.model.critic_optimizers)
            critic_opts = []
            if hasattr(self.model, 'critic_optimizer'):
                critic_opts = [self.model.critic_optimizer]
            elif hasattr(self.model, 'critic_optimizers'):
                critic_opts = list(self.model.critic_optimizers)
            for opt in critic_opts:
                if opt is not None:
                    optimizers.append(opt)
            # Attach schedulers
            self.schedulers = [self.scheduler_factory(opt) for opt in optimizers]
            self.attached = True
        for scheduler in self.schedulers:
            scheduler.step()
        return True

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
        "n_envs": 128,
        "batch": 256,  # Increased batch size for stability
        "grad_steps": 8,
        "net_arch": dict(pi=[32, 32], qf=[32, 32]),
        "buffer": 100_000,
    },
    "MountainCarContinuous-v0": {
        "total_steps": 300_000,
        "n_envs":128,
        "batch": 256,
        "grad_steps": 8,
        "net_arch": dict(pi=[32, 32], qf=[32, 32]),
        "buffer": 100_000,
    },
    "BipedalWalker-v3": {
        "total_steps": 600_000,
        "n_envs": 128,
        "batch": 256,
        "grad_steps": 8,
        "net_arch": dict(pi=[64, 64], qf=[128, 128]),
        "buffer": 1_000_000,
    },
}

TABLE_B = {
    env: {
        "total_steps": 2_000_000 if env != "MountainCarContinuous-v0" else 1_000_000,
        "n_envs": 128,
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
def make_norm_env(env_id, n_envs, clip_reward=1.0):
    venv = make_vec_env(env_id, n_envs=n_envs, wrapper_class=TimeFeatureWrapper)
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=clip_reward
    )
    return venv

def run(env_id: str, table_cfg: dict, extractor_mode: str):
    cfg = table_cfg[env_id]
    # Use a single VecNormalize for both obs and reward
    clip_reward = 10.0 if env_id == "BipedalWalker-v3" else 1.0
    env = make_norm_env(env_id, cfg["n_envs"], clip_reward=clip_reward)
    obs_dim = env.observation_space.shape[0]

    # -------- feature extractor selection --------
    feat_cls, feat_kwargs = None, {}
    if extractor_mode != "Baseline":
        # Dynamically set feature extractor parameters based on obs_dim
        embed = max(32, obs_dim * 4)
        heads = 2 if obs_dim < 8 else 4
        layers = 2 if obs_dim < 8 else 4
        if extractor_mode == "MHA":
            feat_cls, feat_kwargs = MHAFeaturesExtractor, dict(embed_dim=embed, num_heads=heads, ff_hidden=embed*4, num_layers=layers, dropout=0.05, seq_len=obs_dim)
        elif extractor_mode == "LMA":
            d_new = embed // 2
            num_heads_latent = heads
            # Special check for LMA: ensure d_new is a multiple of num_heads_latent
            # NOTE: This needs to be put in a helper function for readability
            max_attempts = 10
            attempt = 0
            while d_new % num_heads_latent != 0 and attempt < max_attempts:
                num_heads_latent = find_closest_divisor(d_new, num_heads_latent)
                attempt += 1
            feat_cls, feat_kwargs = LMAFeaturesExtractor, dict(embed_dim=embed, num_heads_stacking=heads, target_l_new=obs_dim//2, d_new=d_new, num_heads_latent=num_heads_latent, ff_latent_hidden=embed*2, num_lma_layers=layers, dropout=0.05, bias=True, seq_len=obs_dim)
        elif extractor_mode == "MHA_Lite":
            # Special check for MHA Lite: ensure d_new is a multiple of num_heads_latent
            d_new = embed // 2
            max_attempts = 10
            attempt = 0
            while d_new % heads != 0 and attempt < max_attempts:
                heads = find_closest_divisor(d_new, heads)
                attempt += 1
            feat_cls, feat_kwargs = MHAFeaturesExtractor, dict(embed_dim=d_new, num_heads=heads, ff_hidden=d_new*4, num_layers=layers, dropout=0.05, seq_len=obs_dim)

    # Set up policy network architecture and feature extractor
    policy_kwargs = dict(net_arch=cfg["net_arch"])
    if extractor_mode == "Baseline":
        policy_kwargs.update(log_std_init=-0.5)
    else:
        policy_kwargs.update(log_std_init=-2.0)
    if feat_cls:
        policy_kwargs.update(features_extractor_class=SafeFeaturesExtractor, features_extractor_kwargs=dict(extractor_cls=feat_cls, **feat_kwargs))

    if extractor_mode == "MHA":
        lr = 3e-4
        tau = 0.005
        scheduler_factory = lambda optimizer: LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=cfg["total_steps"])
    elif extractor_mode == "LMA":
        lr = 3e-4
        tau = 0.002
        scheduler_factory = lambda optimizer: LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=cfg["total_steps"])
    elif extractor_mode == "MHA_Lite":
        lr = 3e-4
        tau = 0.002
        scheduler_factory = lambda optimizer: LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=cfg["total_steps"])
    else:  # Baseline
        lr = 3e-4
        tau = 0.005
        scheduler_factory = lambda optimizer: LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=cfg["total_steps"])

    lr_scheduler_callback = LRSchedulerCallback(scheduler_factory)

    if extractor_mode == "Baseline":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=cfg['buffer'],
            batch_size=cfg["batch"],
            learning_starts=cfg["batch"] * 4,
            ent_coef="auto",
            tau=tau,
            train_freq=(1, "step"),
            gradient_steps=cfg["grad_steps"],
            policy_kwargs=policy_kwargs,
            device="cuda" if th.cuda.is_available() else "cpu",
            verbose=1,
        )
    else:
        batch_size = 256 if table_cfg is TABLE_A else 1024
        if "max_grad_norm" in policy_kwargs:
            del policy_kwargs["max_grad_norm"]
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=cfg["buffer"],
            batch_size=batch_size,
            learning_starts=batch_size * 4,
            ent_coef="auto",
            tau=tau,
            train_freq=(1, "step"),
            gradient_steps=cfg["grad_steps"],
            policy_kwargs=policy_kwargs,
            device="cuda" if th.cuda.is_available() else "cpu",
            verbose=1,
        )

    # Train the agent
    callbacks = [ClipGradCallback(max_norm=0.5), lr_scheduler_callback]
    model.learn(total_timesteps=cfg["total_steps"], progress_bar=True, callback=callbacks)

    # Save VecNormalize stats after training
    env.save("vecnorm.pkl")

    # ------------------- deterministic evaluation -------------------
    eval_env_raw = make_vec_env(env_id, n_envs=1, wrapper_class=TimeFeatureWrapper)
    eval_env     = VecNormalize.load("vecnorm.pkl", eval_env_raw)
    eval_env.training = False
    eval_env.norm_reward = False
    rets = []
    for _ in range(100):
        obs = eval_env.reset()
        done, ep_ret = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(act)
            ep_ret += reward
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
        for extractor in ["Baseline", "MHA", "LMA", "MHA_Lite"]:
        #for extractor in ["MHA", "LMA", "MHA_Lite"]: # Debugging Specific Feature Extractors
            run(env, table, extractor)
