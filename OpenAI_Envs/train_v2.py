import os
os.environ["TQDM_DISABLE_RICH"] = "1"

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import warnings
from classes import *  # custom extractors
from tqdm_utils import suppress_tqdm_cleanup

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 0.  numpy compat shim
# -----------------------------------------------------------------------------
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "float_"):
    np.float_ = np.float64

# -----------------------------------------------------------------------------
# 1.  Table‑specific hyper‑params (A = quick, B = full)
# -----------------------------------------------------------------------------
TABLE_A = {
    "Pendulum-v1": {
        "total_steps": 300_000,
        "n_envs": 8,
        "batch": 64,
        "grad_steps": 8,
        "lr": 1e-3,
        "tau": 0.02,
        "net_arch": dict(pi=[32, 32], qf=[32, 32]),
        "buffer": 100_000,
    },
    "MountainCarContinuous-v0": {
        "total_steps": 300_000,
        "n_envs": 8,
        "batch": 64,
        "grad_steps": 8,
        "lr": 1e-3,
        "tau": 0.02,
        "net_arch": dict(pi=[32, 32], qf=[32, 32]),
        "buffer": 100_000,
    },
    "BipedalWalker-v3": {
        "total_steps": 600_000,
        "n_envs": 8,
        "batch": 128,
        "grad_steps": 8,
        "lr": 3e-4,
        "tau": 0.02,
        "net_arch": dict(pi=[64, 64], qf=[128, 128]),
        "buffer": 1_000_000,
    },
}

TABLE_B = {
    env: {
        "total_steps": 2_000_000 if env != "MountainCarContinuous-v0" else 1_000_000,
        "n_envs": 64,
        "batch": 1_024,
        "grad_steps": 64,
        "lr": 3e-4,
        "tau": 0.005,
        "net_arch": dict(pi=[128, 128], qf=[256, 256]),
        "buffer": 1_000_000,
    }
    for env in ["Pendulum-v1", "MountainCarContinuous-v0", "BipedalWalker-v3"]
}

# -----------------------------------------------------------------------------
# 2.  Safe wrapper for any extractor
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
# -----------------------------------------------------------------------------

def run(env_id: str, table_cfg: dict, extractor_mode: str):
    cfg = table_cfg[env_id]
    env = make_vec_env(env_id, n_envs=cfg["n_envs"])
    obs_dim = env.observation_space.shape[0]

    # -------- feature extractor --------
    feat_cls, feat_kwargs = None, {}
    if extractor_mode != "Baseline":
        embed = max(32, obs_dim * 4)
        heads = 2 if obs_dim < 8 else 4
        layers = 2 if obs_dim < 8 else 4
        if extractor_mode == "MHA":
            feat_cls, feat_kwargs = MHAFeaturesExtractor, dict(embed_dim=embed, num_heads=heads, ff_hidden=embed*4, num_layers=layers, dropout=0.05, seq_len=obs_dim)
        elif extractor_mode == "LMA":
            feat_cls, feat_kwargs = LMAFeaturesExtractor, dict(embed_dim=embed, num_heads_stacking=heads, target_l_new=obs_dim//2, d_new=embed//2, num_heads_latent=heads, ff_latent_hidden=embed*2, num_lma_layers=layers, dropout=0.05, bias=True, seq_len=obs_dim)
        elif extractor_mode == "MHA_Lite":
            feat_cls, feat_kwargs = MHAFeaturesExtractor, dict(embed_dim=embed//2, num_heads=heads, ff_hidden=(embed//2)*4, num_layers=layers, dropout=0.05, seq_len=obs_dim)

    policy_kwargs = dict(net_arch=cfg["net_arch"])
    if feat_cls:
        policy_kwargs.update(features_extractor_class=SafeFeaturesExtractor, features_extractor_kwargs=dict(extractor_cls=feat_cls, **feat_kwargs))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg["lr"],
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
    )

    model.learn(total_timesteps=cfg["total_steps"], progress_bar=True)

    # deterministic eval
    eval_env = gym.make(env_id)
    rets = []
    for _ in range(50):
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
    with open("results.log", "a") as f:
        f.write(log_line)

# -----------------------------------------------------------------------------
# 4.  entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mode = input("Select table (A=quick / B=full) [A]: ").strip().lower() or "a"
    assert mode in ("a", "b"), "Please type 'A' or 'B'"
    table = TABLE_A if mode == "a" else TABLE_B

    for env in table.keys():
        for extractor in ["Baseline", "MHA", "LMA", "MHA_Lite"]:
            run(env, table, extractor)
