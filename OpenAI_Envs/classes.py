# classes.py
# This is the stable-baselines3 implementation of the LMA feature extractor. It is quite unstable so we decided to use another library instead


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass, field
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback


#===============================================
# MHA Feature Extractor Implementation
#===============================================

class MHAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        embed_dim=64,
        num_heads=3,
        ff_hidden=128,
        num_layers=4,
        seq_len=6,
        dropout=0.3
    ):
        """
        :param observation_space: Gym observation space.
        :param embed_dim: Size of the embedding (d_model) in the Transformer.
        :param num_heads: Number of attention heads in the multi-head attention layers.
        :param ff_hidden: Dimension of the feedforward network in the Transformer.
        :param num_layers: Number of layers in the Transformer encoder.
        :param seq_len: Number of time steps to unroll in the Transformer.
        :param dropout: Dropout probability to use throughout the model.
        """
        # Features dimension after Transformer = embed_dim * seq_len
        feature_dim = embed_dim * seq_len
        super(MHAFeaturesExtractor, self).__init__(observation_space, features_dim=feature_dim)

        self.embed_dim = embed_dim
        self.input_dim = observation_space.shape[0]
        self.seq_len = seq_len  
        self.dropout_p = dropout

        # Validate that seq_len divides input_dim evenly
        if self.input_dim % seq_len != 0:
            raise ValueError("Input dimension must be divisible by seq_len.")

        # Linear projection for input -> embedding
        self.input_embedding = nn.Linear(self.input_dim // seq_len, embed_dim)

        # Dropout layer for embeddings
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)

        # Transformer Encoder
        #   - The 'dropout' parameter here applies to:
        #       1) The self-attention mechanism outputs
        #       2) The feed-forward sub-layer outputs
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=self.dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Flatten final output to feed into the policy & value networks
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Reshape input into (batch_size, seq_len, features_per_seq)
        batch_size = x.shape[0]
        features_per_seq = self.input_dim // self.seq_len
        x = x.view(batch_size, self.seq_len, features_per_seq)

        # Linear projection
        x = self.input_embedding(x)

        # Add positional encoding
        batch_size, seq_len, embed_dim = x.shape
        x = x + self._positional_encoding(seq_len, embed_dim).to(x.device)

        # Drop out some embeddings for regularization
        x = self.embedding_dropout(x)

        # Pass sequence through the Transformer encoder
        x = self.transformer(x)

        # Flatten for final feature vector
        return self.flatten(x)

    def _positional_encoding(self, seq_len, embed_dim):
        """Sine-cosine positional encoding, shape: (seq_len, embed_dim)."""
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # The standard “div_term” for sine/cosine in attention
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
#===============================================
# LMA Helper Function
#===============================================
def find_closest_divisor(total_value, target_divisor, max_delta=100):
    """ Finds closest divisor. """
    if not isinstance(total_value, int) or total_value <= 0: raise ValueError(f"total_value ({total_value}) must be positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0: target_divisor = max(1, target_divisor)
    if not isinstance(max_delta, int) or max_delta < 0: raise ValueError(f"max_delta ({max_delta}) must be non-negative.")
    if total_value == 0: return 1
    if target_divisor > 0 and total_value % target_divisor == 0: return target_divisor
    search_start = max(1, target_divisor)
    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0: return candidate_minus
        candidate_plus = search_start + delta
        if candidate_plus > 0 and total_value % candidate_plus == 0: return candidate_plus
    # Fallback if no close divisor found
    print(f"Warning: No divisor found near {target_divisor} for {total_value}. Searching all divisors.")
    best_divisor = 1
    min_diff = abs(target_divisor - 1)
    for i in range(2, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0:
            div1 = i
            div2 = total_value // i
            diff1 = abs(target_divisor - div1)
            diff2 = abs(target_divisor - div2)
            if diff1 < min_diff:
                min_diff = diff1
                best_divisor = div1
            if diff2 < min_diff:
                min_diff = diff2
                best_divisor = div2
    # Check total_value itself
    diff_total = abs(target_divisor - total_value)
    if diff_total < min_diff:
        best_divisor = total_value

    print(f"Using {best_divisor} as fallback divisor.")
    return best_divisor

#===============================================
# LMA Feature Extractor Implementation
#===============================================

@dataclass
class LMAConfigRL:
    """ Config for LMA Feature Extractor """
    seq_len: int             # Input sequence length (L, e.g., 6)
    embed_dim: int           # Initial embedding dim (d0)
    num_heads_stacking: int  # Heads for stacking (nh)
    target_l_new: int        # Target latent sequence length
    d_new: int               # Latent embedding dim
    num_heads_latent: int    # Heads for latent attention

    # Derived values
    L_new: int = field(init=False) # Actual latent sequence length
    C_new: int = field(init=False) # Latent chunk size

    def __post_init__(self):
        if self.seq_len <= 0 or self.embed_dim <= 0 or self.num_heads_stacking <= 0 or \
           self.target_l_new <= 0 or self.d_new <= 0 or self.num_heads_latent <= 0:
            raise ValueError("LMAConfigRL inputs must be positive.")
        if self.embed_dim % self.num_heads_stacking != 0:
            raise ValueError(f"LMA embed_dim ({self.embed_dim}) not divisible by num_heads_stacking ({self.num_heads_stacking})")
        if self.d_new % self.num_heads_latent != 0:
            raise ValueError(f"LMA d_new ({self.d_new}) not divisible by num_heads_latent ({self.num_heads_latent})")

        total_features = self.seq_len * self.embed_dim
        if total_features == 0: raise ValueError("LMA total features cannot be zero.")

        try:
            self.L_new = find_closest_divisor(total_features, self.target_l_new)
            if self.L_new != self.target_l_new:
                 print(f"LMAConfigRL ADJUSTMENT: L_new {self.target_l_new} -> {self.L_new}")
            if self.L_new <= 0: raise ValueError("Calculated L_new is not positive.")
            if total_features % self.L_new != 0:
                raise RuntimeError(f"Internal Error: total_features ({total_features}) not divisible by final L_new ({self.L_new})")
            self.C_new = total_features // self.L_new
            if self.C_new <= 0: raise ValueError("Calculated C_new is not positive.")
        except ValueError as e:
            raise ValueError(f"LMA Config Error calculating L_new/C_new: {e}") from e

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LMA_InitialTransform_RL(nn.Module):
    """ Performs LMA Stage 1 and Stage 2 (Stacking, Rechunking, Latent Embed) """
    def __init__(self, features_per_step: int, lma_config: LMAConfigRL, dropout: float, bias: bool):
        super().__init__()
        self.lma_config = lma_config
        self.dropout_p = dropout
        self.bias = bias

        # Stage 1 equivalent: Project features per step to embed_dim (d0)
        self.input_embedding = nn.Linear(features_per_step, lma_config.embed_dim, bias=self.bias)
        self.input_embedding_act = nn.ReLU() # Activation defined
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)

        # Stage 2b: Latent Embedding Layer (maps C_new -> d_new)
        self.embed_layer_2 = nn.Linear(lma_config.C_new, lma_config.d_new, bias=self.bias)
        # Using ReLU here as requested (instead of GELU in prev version)
        self.embed_layer_2_act = nn.ReLU()

        print("  LMA_InitialTransform_RL Initialized:")
        print(f"    Input features/step: {features_per_step}")
        print(f"    Stage 1 Projection: Linear({features_per_step} -> {lma_config.embed_dim}) + ReLU") # Updated print
        print(f"    Head Stacking: {lma_config.num_heads_stacking} heads")
        print(f"    Rechunking: L={lma_config.seq_len}, d0={lma_config.embed_dim} -> L_new={lma_config.L_new}, C_new={lma_config.C_new}")
        print(f"    Stage 2b Projection: Linear({lma_config.C_new} -> {lma_config.d_new}) + ReLU") # Updated print

    def forward(self, x):
        # Input x shape: (B, L, features_per_step)
        B, L, _ = x.shape
        if L != self.lma_config.seq_len:
            raise ValueError(f"Input sequence length {L} doesn't match LMA config seq_len {self.lma_config.seq_len}")

        # --- Stage 1 ---
        y = self.input_embedding(x) # (B, L, Feat/L) -> (B, L, d0)
        y = self.input_embedding_act(y) # *** APPLY ACTIVATION HERE ***
        y = y + self._positional_encoding(L, self.lma_config.embed_dim).to(y.device)
        y = self.embedding_dropout(y) # (B, L, d0)

        # --- Stage 2a: Head-View Stacking ---
        d0 = self.lma_config.embed_dim
        nh = self.lma_config.num_heads_stacking
        dk = d0 // nh
        try:
            head_views = torch.split(y, dk, dim=2)
            x_stacked = torch.cat(head_views, dim=1) # (B, L*nh, dk)
        except Exception as e:
            raise RuntimeError(f"Error during head stacking: Input={y.shape}, d0={d0}, nh={nh}, dk={dk}") from e

        # --- Stage 2b: Re-Chunking & Latent Embedding ---
        L_new = self.lma_config.L_new
        C_new = self.lma_config.C_new
        expected_flat_dim = L * d0

        x_flat = x_stacked.view(B, -1) # (B, L*d0)
        if x_flat.shape[1] != expected_flat_dim:
             raise RuntimeError(f"Flattened shape mismatch: Expected {expected_flat_dim}, got {x_flat.shape[1]}")

        try: x_rechunked = x_flat.view(B, L_new, C_new) # (B, L_new, C_new)
        except RuntimeError as e: raise RuntimeError(f"Error rechunking: Flat={x_flat.shape}, Target=({B}, {L_new}, {C_new})") from e

        z_embedded = self.embed_layer_2(x_rechunked) # (B, L_new, d_new)
        z = self.embed_layer_2_act(z_embedded) # Apply activation

        return z # Return latent representation

    def _positional_encoding(self, seq_len, embed_dim):
        # (Keep positional encoding function as before)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class LatentAttention_RL(nn.Module):
    """ MHA operating in the LMA latent space (Non-Causal) """
    def __init__(self, d_new: int, num_heads_latent: int, dropout: float, bias: bool):
        super().__init__()
        assert d_new % num_heads_latent == 0
        self.d_new = d_new
        self.num_heads = num_heads_latent
        self.head_dim = d_new // num_heads_latent
        self.dropout_p = dropout
        self.bias = bias

        self.c_attn = nn.Linear(d_new, 3 * d_new, bias=self.bias)
        self.c_proj = nn.Linear(d_new, d_new, bias=self.bias)
        self.attn_dropout = nn.Dropout(self.dropout_p)
        self.resid_dropout = nn.Dropout(self.dropout_p)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if self.flash: print(f"    - LatentAttention_RL: Using Flash Attention (d_new={d_new})")
        else: print(f"    - LatentAttention_RL: Using slow attention path.")

    def forward(self, z):
        B, L_new, C = z.size()
        if C != self.d_new: raise ValueError(f"LatentAttention C mismatch")
        q, k, v = self.c_attn(z).split(self.d_new, dim=2)
        q = q.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L_new, self.d_new)
        y = self.resid_dropout(self.c_proj(y))
        return y

class LatentMLP_RL(nn.Module):
    """ MLP operating in the latent space dimension d_new """
    def __init__(self, d_new: int, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc    = nn.Linear(d_new, ff_latent_hidden, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(ff_latent_hidden, d_new, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.dropout(x)
        return x

class LMABlock_RL(nn.Module):
    """ A single LMA block operating in the latent space """
    def __init__(self, lma_config: LMAConfigRL, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__()
        self.ln_1 = LayerNorm(lma_config.d_new, bias=bias)
        self.attn = LatentAttention_RL(lma_config.d_new, lma_config.num_heads_latent, dropout, bias)
        self.ln_2 = LayerNorm(lma_config.d_new, bias=bias)
        self.mlp = LatentMLP_RL(lma_config.d_new, ff_latent_hidden, dropout, bias)

    def forward(self, z):
        z = z + self.attn(self.ln_1(z))
        z = z + self.mlp(self.ln_2(z))
        return z

class LMAFeaturesExtractor(BaseFeaturesExtractor):
    """ Feature extractor using the original LMA mechanism """
    def __init__(
        self,
        observation_space,
        embed_dim=64, num_heads_stacking=4, target_l_new=3, d_new=32,
        num_heads_latent=4, ff_latent_hidden=64, num_lma_layers=2,
        seq_len=4, dropout=0.1, bias=True
    ):
        print("\n--- Initializing LMAFeaturesExtractor ---")
        print("Calculating LMA dimensions...")
        self.lma_config = LMAConfigRL(
            seq_len=seq_len, embed_dim=embed_dim, num_heads_stacking=num_heads_stacking,
            target_l_new=target_l_new, d_new=d_new, num_heads_latent=num_heads_latent
        )
        print(f"  Final LMA Config: L={self.lma_config.seq_len}, d0={self.lma_config.embed_dim}, nh_stack={self.lma_config.num_heads_stacking}")
        print(f"                    L_new={self.lma_config.L_new}, C_new={self.lma_config.C_new}, d_new={self.lma_config.d_new}, nh_latent={self.lma_config.num_heads_latent}")

        feature_dim = self.lma_config.L_new * self.lma_config.d_new
        super().__init__(observation_space, features_dim=feature_dim)
        print(f"  SB3 features_dim (Flattened L_new * d_new): {feature_dim}")

        self.input_dim_total = observation_space.shape[0]
        self.seq_len = seq_len
        if self.input_dim_total % seq_len != 0:
            raise ValueError(f"Input dimension ({self.input_dim_total}) must be divisible by seq_len ({seq_len}).")
        self.features_per_step = self.input_dim_total // seq_len

        self.initial_transform = LMA_InitialTransform_RL(
            features_per_step=self.features_per_step,
            lma_config=self.lma_config, dropout=dropout, bias=bias
        )
        self.lma_blocks = nn.ModuleList([
            LMABlock_RL(
                lma_config=self.lma_config, ff_latent_hidden=ff_latent_hidden,
                dropout=dropout, bias=bias
            ) for _ in range(num_lma_layers)
        ])
        print(f"  Number of LMA Blocks: {num_lma_layers}")
        self.flatten = nn.Flatten()
        print("-----------------------------------------")

    def forward(self, x):
        batch_size = x.shape[0]
        try:
             x_reshaped = x.view(batch_size, self.seq_len, self.features_per_step)
        except RuntimeError as e:
             raise RuntimeError(f"Error reshaping input: Input={x.shape}, Target=({batch_size},{self.seq_len},{self.features_per_step})") from e
        z = self.initial_transform(x_reshaped)
        for block in self.lma_blocks:
            z = block(z)
        features = self.flatten(z)
        return features

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check each info dict for episode information
        for info in self.locals.get("infos", []):
            if 'episode' in info:
                # Append the total reward of the finished episode
                self.episode_rewards.append(info['episode']['r'])
        return True

class LearningRateScheduler(BaseCallback):
    def __init__(self, initial_lr: float, final_lr: float, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps
        print(f"Initializing LR scheduler: {initial_lr:.2e} -> {final_lr:.2e}")
    
    def _on_step(self) -> bool:
        progress_fraction = self.num_timesteps / self.total_timesteps
        current_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress_fraction
        
        # Update both the model's learning rate and optimizer's learning rate
        self.model.learning_rate = current_lr
        optimizer = self.model.policy.optimizer
        
        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        # Verify both learning rates
        actual_lr = optimizer.param_groups[0]['lr']
        
        # Log every 10000 steps
        if self.num_timesteps % 10000 == 0:
            print(f"Step {self.num_timesteps}/{self.total_timesteps}")
            print(f"Target LR = {current_lr:.2e}, Actual LR = {actual_lr:.2e}")
            print(f"Model LR = {self.model.learning_rate:.2e}")
        
        return True

    def _on_training_start(self) -> None:
        """Called before training starts"""
        print("LR Scheduler: Training starting...")
        print(f"Initial learning rate: {self.model.learning_rate:.2e}")
        
class GradientMonitorCallback(BaseCallback):
    def _on_step(self) -> bool:
        for name, param in self.model.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 10:  # Threshold for exploding gradients
                    print(f"Warning: High gradient norm ({grad_norm:.2f}) in {name}")
        return True