# classes.py

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

# Internal Module Imports / Default Values
try:
    import globals
    step_size = getattr(globals, 'step_size', 10)
    spawn_range = getattr(globals, 'spawn_range', 50)
    grid_size = getattr(globals, 'grid_size', 500)
    POSITIONAL_DEBUG = getattr(globals, 'POSITIONAL_DEBUG', False)
    REWARD_DEBUG = getattr(globals, 'REWARD_DEBUG', False)
except ImportError:
    print("Warning: 'globals.py' not found. Using default values.")
    step_size = 10
    spawn_range = 50
    grid_size = 500
    POSITIONAL_DEBUG = False
    REWARD_DEBUG = False

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
    print(f"Warning: No divisor found near {target_divisor} for {total_value}. Searching all divisors.")
    best_divisor = 1; min_diff = abs(target_divisor - 1)
    for i in range(2, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0:
            div1 = i; div2 = total_value // i
            diff1 = abs(target_divisor - div1); diff2 = abs(target_divisor - div2)
            if diff1 < min_diff: min_diff = diff1; best_divisor = div1
            if diff2 < min_diff: min_diff = diff2; best_divisor = div2
    diff_total = abs(target_divisor - total_value)
    if diff_total < min_diff: best_divisor = total_value
    print(f"Using {best_divisor} as fallback divisor.")
    return best_divisor

#===============================================
# Simulation Objects
#===============================================
class BaseObject:
    def __init__(self, name, initial_position, color='blue'):
        self.name = name
        self.position = np.array(initial_position, dtype=float)
        self.path = [self.position.copy()]
        self.color = color

    def move(self, movement_vector):
        self.position += np.array(movement_vector, dtype=float)
        self.path.append(self.position.copy())

    def set_position(self, new_position):
        self.position = np.array(new_position, dtype=float)
        self.path.append(self.position.copy())

class CCA(BaseObject):
    def __init__(self, name, initial_position):
        super().__init__(name, initial_position, color='green')

class Foxtrot(BaseObject):
    """
    Foxtrot object that can move autonomously in a cube pattern.
    Manages its own movement state internally.
    """
    def __init__(self, name, initial_position, speed=None, grid_size_param=None):
        """
        Initializes the Foxtrot object.

        Args:
            name (str): Name of the object.
            initial_position (list or np.ndarray): Starting position [x, y, z].
            speed (float, optional): Movement speed per step. Defaults to globals.step_size or 10.
            grid_size_param (int, optional): Size of the grid boundary. Defaults to globals.grid_size or 500.
        """
        super().__init__(name, initial_position, color='orange')

        # Use provided params or get from globals/defaults
        self.grid_size = grid_size_param if grid_size_param is not None else getattr(globals, 'grid_size', 500)
        default_step = getattr(globals, 'step_size', 10)
        self.speed = float(speed if speed is not None else default_step)

        # Internal state for cube movement pattern
        self._initialize_movement_state()
        # Optional: Add a print statement for debugging initialization
        # if POSITIONAL_DEBUG: print(f"Foxtrot '{self.name}' initialized. Speed: {self.speed}. Grid: {self.grid_size}")

    def _initialize_movement_state(self):
        """Resets the internal state dictionary for the cube movement pattern."""
        self.cube_state = {"phase": "rectangle_init", "rectangle_info": {}, "z_info": {}}
        # Optional: Debug print for state reset
        # if POSITIONAL_DEBUG: print(f"  Movement state reset for {self.name}. Phase: {self.cube_state['phase']}")

    def set_position(self, new_position):
        """
        Sets a new position for the Foxtrot and resets the cube movement state.
        Overrides BaseObject.set_position to add state reset.
        """
        super().set_position(new_position) # Call parent method to update position and path
        self._initialize_movement_state() # Reset the movement phase/state

    def move_cube(self, grid_size=None, step_size=None):
        """
        Moves the Foxtrot one step according to the cube (rectangle + Z) pattern.
        Uses internal state (self.cube_state) and speed (self.speed).
        Updates self.position and appends to self.path.

        Args:
            grid_size (int, optional): Overrides the instance's grid_size for this step's
                                       clipping/randomization. Defaults to self.grid_size.
            step_size (float, optional): Overrides the instance's speed for this step.
                                         Defaults to self.speed.
        """
        # Use provided args or instance defaults
        current_grid_size = grid_size if grid_size is not None else self.grid_size
        # NOTE: Renamed internal variable to avoid confusion with the parameter name
        movement_speed = step_size if step_size is not None else self.speed

        # Ensure state is initialized (safety check)
        if not hasattr(self, 'cube_state') or not self.cube_state:
             print(f"Warning: cube_state missing for {self.name}. Re-initializing.")
             self._initialize_movement_state()

        # --- Cube Movement Logic ---
        pos_float = self.position.astype(float) # Work with a float copy
        cube_state = self.cube_state
        phase = cube_state.get("phase", "rectangle_init")

        # --- Rectangle Phase ---
        if phase.startswith("rectangle"):
            rect_info = cube_state.setdefault("rectangle_info", {})

            if phase == "rectangle_init":
                center_xy = pos_float[:2]
                # Use instance grid size for random range if not overridden? Or always current? Let's use current.
                length = np.random.uniform(50, min(200, current_grid_size*0.4)) # Limit size relative to grid
                width = np.random.uniform(50, min(200, current_grid_size*0.4))
                half_l, half_w = length / 2.0, width / 2.0
                v0=center_xy+np.array([+half_l,+half_w]); v1=center_xy+np.array([-half_l,+half_w])
                v2=center_xy+np.array([-half_l,-half_w]); v3=center_xy+np.array([+half_l,-half_w])
                rect_info["vertices"]=[v0,v1,v2,v3]; rect_info["current_target_idx"]=0
                rect_info["laps_completed"]=0; cube_state["phase"]="rectangle_move"
                #if POSITIONAL_DEBUG: print(f"  {self.name}: Init Rect @ {center_xy.round(1)}. Phase -> {cube_state['phase']}")
                # No actual position change in init step

            elif phase == "rectangle_move":
                target_idx = rect_info.get("current_target_idx", 0); vertices = rect_info.get("vertices")
                if not vertices: # Safety check
                    cube_state["phase"] = "rectangle_init"; print(f"Warning: Rect vertices missing for {self.name}, resetting phase.")
                else:
                    target_xy=vertices[target_idx]; current_xy=pos_float[:2]
                    direction_xy=target_xy-current_xy; dist_xy=np.linalg.norm(direction_xy)
                    if dist_xy < movement_speed: # Reached target vertex
                        pos_float[:2]=target_xy # Snap to vertex
                        next_target_idx=(target_idx+1)%len(vertices); rect_info["current_target_idx"]=next_target_idx
                        if next_target_idx == 0: # Completed a lap
                            laps = rect_info.get("laps_completed", 0) + 1; rect_info["laps_completed"] = laps
                            #if POSITIONAL_DEBUG: print(f"  {self.name}: Lap {laps} complete.")
                            if laps >= 1: # Move to Z phase after 1 lap
                                cube_state["phase"] = "z_move_init";
                                #if POSITIONAL_DEBUG: print(f"  {self.name}: Rect done. Phase -> {cube_state['phase']}")
                    else: # Move towards target vertex
                        step_vector_xy=(direction_xy/dist_xy)*movement_speed; pos_float[:2]+=step_vector_xy

        # --- Z Phase ---
        elif phase.startswith("z_move"):
            z_info = cube_state.setdefault("z_info", {})

            if phase == "z_move_init":
                # Randomize target Z using the potentially overridden grid size
                z_info["target_z"]=np.random.uniform(0, current_grid_size); cube_state["phase"]="z_move"
                #if POSITIONAL_DEBUG: print(f"  {self.name}: Init Z. Target: {z_info['target_z']:.1f}. Phase -> {cube_state['phase']}")
                # No actual position change in init step

            elif phase == "z_move":
                target_z = z_info.get("target_z")
                if target_z is None: # Safety check
                    cube_state["phase"] = "z_move_init"; print(f"Warning: Target Z missing for {self.name}, resetting phase.")
                else:
                    current_z=pos_float[2]; delta_z=target_z-current_z
                    if abs(delta_z)<movement_speed: # Reached target Z
                        pos_float[2]=target_z # Snap to target
                        cube_state["phase"]="rectangle_init" # Go back to rectangle phase
                        #if POSITIONAL_DEBUG: print(f"  {self.name}: Target Z reached. Phase -> {cube_state['phase']}")
                    else: # Move towards target Z
                        pos_float[2]+=np.sign(delta_z)*movement_speed

        # --- Update Position and Path ---
        # Clip final calculated position using the potentially overridden grid size
        self.position = np.clip(pos_float, 0, current_grid_size - 1)
        # Manually append to path since we didn't call BaseObject.move
        self.path.append(self.position.copy())

#===============================================
# Simulation Engine (Visualization - unchanged logic)
#===============================================
class SimulationEngine:
    def __init__(self, grid_size_param=1000):
        self.grid_size = grid_size_param
        self.objects = []
        self.fig = None
        self.ax = None
        self.anim = None

    def add_object(self, obj):
        if self.is_within_bounds(obj.position):
            self.objects.append(obj)
        else: print(f"Object {obj.name} initial position {obj.position} OOB!")

    def is_within_bounds(self, position):
        return all(0 <= pos < self.grid_size for pos in position)

    def simulate(self, steps, movement_logic_fn):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._setup_ax()
        self.anim = FuncAnimation(self.fig, self.update_frame, frames=steps,
                                  fargs=(movement_logic_fn,), interval=100, repeat=False)
        plt.show()

    def _setup_ax(self):
        self.ax.set_xlim([0, self.grid_size]); self.ax.set_ylim([0, self.grid_size]); self.ax.set_zlim([0, self.grid_size])
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title('Simulation Visualization')

    def update_frame(self, frame, movement_logic_fn):
        self.ax.cla(); self._setup_ax()
        movement_vectors = movement_logic_fn(frame)
        plotted_objects = {}
        for obj in self.objects:
            movement_vector = movement_vectors.get(obj.name, np.zeros_like(obj.position))
            obj.move(movement_vector)
            if not self.is_within_bounds(obj.position): print(f"{obj.name} OOB {obj.position} step {frame}!")
            if len(obj.path) > 1:
                path_array = np.array(obj.path)
                label_path = f'{obj.name} Path' if obj.name not in plotted_objects else None
                self.ax.plot(path_array[:,0], path_array[:,1], path_array[:,2], label=label_path, color=obj.color, alpha=0.5)
            label_pos = obj.name if obj.name not in plotted_objects else None
            self.ax.scatter(*obj.position, color=obj.color, label=label_pos, s=100, depthshade=True)
            plotted_objects[obj.name] = True
        if plotted_objects: self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()

#===============================================
# Movement Functions (If Foxtrot needs to move later)
#===============================================
def foxtrot_movement_fn(position):
    direction_probabilities = [0.4, 0.4, 0.2]; axes_to_move = np.random.choice([0,1,2], size=np.random.randint(1,4), replace=False, p=np.array(direction_probabilities)/sum(direction_probabilities))
    random_step = np.zeros(3);
    for axis in axes_to_move: random_step[axis] = np.random.uniform(-step_size, step_size)
    new_position = position + random_step; new_position = np.clip(new_position, 0, grid_size - 1); return new_position
def foxtrot_movement_fn_cube(position, cube_state):
    pos_float=position.astype(float)
    if"phase"not in cube_state:cube_state["phase"]="rectangle_init";cube_state["rectangle_info"]={};cube_state["z_info"]={}
    if cube_state["phase"].startswith("rectangle"):
        rect_info=cube_state["rectangle_info"]
        if cube_state["phase"]=="rectangle_init":
            length=np.random.uniform(50,200);width=np.random.uniform(50,200);center_xy=pos_float[:2];half_l,half_w=length/2.0,width/2.0
            v0=center_xy+np.array([+half_l,+half_w]);v1=center_xy+np.array([-half_l,+half_w]);v2=center_xy+np.array([-half_l,-half_w]);v3=center_xy+np.array([+half_l,-half_w])
            rect_info["vertices"]=[v0,v1,v2,v3];rect_info["current_target_idx"]=0;rect_info["laps_completed"]=0;cube_state["phase"]="rectangle_move"
        if cube_state["phase"]=="rectangle_move":
            target_idx=rect_info["current_target_idx"];target_xy=rect_info["vertices"][target_idx];current_xy=pos_float[:2];direction_xy=target_xy-current_xy;dist_xy=np.linalg.norm(direction_xy)
            if dist_xy<step_size:
                pos_float[:2]=target_xy;next_target_idx=(target_idx+1)%len(rect_info["vertices"]);rect_info["current_target_idx"]=next_target_idx
                if next_target_idx==0:
                    rect_info["laps_completed"]+=1
                    if rect_info["laps_completed"]>=1:cube_state["phase"]="z_move_init"
            else:step_vector_xy=(direction_xy/dist_xy)*step_size;pos_float[:2]+=step_vector_xy
    elif cube_state["phase"].startswith("z_move"):
        z_info=cube_state["z_info"]
        if cube_state["phase"]=="z_move_init":z_info["target_z"]=np.random.uniform(0,grid_size);cube_state["phase"]="z_move"
        if cube_state["phase"]=="z_move":
            target_z=z_info["target_z"];current_z=pos_float[2];delta_z=target_z-current_z
            if abs(delta_z)<step_size:pos_float[2]=target_z;cube_state["phase"]="rectangle_init"
            else:pos_float[2]+=np.sign(delta_z)*step_size
    pos_float=np.clip(pos_float,0,grid_size-1);return pos_float

#===============================================
# LMA Feature Extractor Implementation
#===============================================

@dataclass
class LMAConfigRL:
    """ Config for LMA Feature Extractor """
    seq_len: int; embed_dim: int; num_heads_stacking: int
    target_l_new: int; d_new: int; num_heads_latent: int
    L_new: int = field(init=False); C_new: int = field(init=False)
    def __post_init__(self):
        if not all(v>0 for v in [self.seq_len, self.embed_dim, self.num_heads_stacking, self.target_l_new, self.d_new, self.num_heads_latent]): raise ValueError("LMAConfigRL inputs must be positive.")
        if self.embed_dim % self.num_heads_stacking != 0: raise ValueError(f"LMA embed_dim ({self.embed_dim}) not divisible by num_heads_stacking ({self.num_heads_stacking})")
        if self.d_new % self.num_heads_latent != 0: raise ValueError(f"LMA d_new ({self.d_new}) not divisible by num_heads_latent ({self.num_heads_latent})")
        total_features = self.seq_len * self.embed_dim
        if total_features == 0: raise ValueError("LMA total features cannot be zero.")
        try:
            self.L_new = find_closest_divisor(total_features, self.target_l_new)
            if self.L_new != self.target_l_new: print(f"LMAConfigRL ADJUSTMENT: L_new {self.target_l_new} -> {self.L_new}")
            if self.L_new <= 0: raise ValueError("Calculated L_new is not positive.")
            if total_features % self.L_new != 0: raise RuntimeError(f"Internal Error: total_features ({total_features}) not divisible by final L_new ({self.L_new})")
            self.C_new = total_features // self.L_new
            if self.C_new <= 0: raise ValueError("Calculated C_new is not positive.")
        except ValueError as e: raise ValueError(f"LMA Config Error calculating L_new/C_new: {e}") from e

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LMA_InitialTransform_RL(nn.Module):
    def __init__(self, features_per_step: int, lma_config: LMAConfigRL, dropout: float, bias: bool):
        super().__init__()
        self.lma_config = lma_config; self.dropout_p = dropout; self.bias = bias
        self.input_embedding = nn.Linear(features_per_step, lma_config.embed_dim, bias=self.bias)
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)
        self.embed_layer_2 = nn.Linear(lma_config.C_new, lma_config.d_new, bias=self.bias)
        self.embed_layer_2_act = nn.GELU()
        print("  LMA_InitialTransform_RL Initialized:"); print(f"    Input feat/step: {features_per_step}, Stage 1 Proj: Lin({features_per_step}->{lma_config.embed_dim})")
        print(f"    Stacking: {lma_config.num_heads_stacking} heads, Rechunk: L={lma_config.seq_len},d0={lma_config.embed_dim}->L_new={lma_config.L_new},C_new={lma_config.C_new}")
        print(f"    Stage 2b Proj: Lin({lma_config.C_new}->{lma_config.d_new})")

    def forward(self, x):
        B, L, _ = x.shape
        if L != self.lma_config.seq_len: raise ValueError(f"Input L {L} != config L {self.lma_config.seq_len}")
        y = self.input_embedding(x); y = y + self._positional_encoding(L, self.lma_config.embed_dim).to(y.device); y = self.embedding_dropout(y)
        d0 = self.lma_config.embed_dim; nh = self.lma_config.num_heads_stacking; dk = d0 // nh
        try: head_views = torch.split(y, dk, dim=2); x_stacked = torch.cat(head_views, dim=1)
        except Exception as e: raise RuntimeError(f"Head stacking error: Input={y.shape}, d0={d0}, nh={nh}, dk={dk}") from e
        L_new=self.lma_config.L_new; C_new=self.lma_config.C_new; expected_flat_dim=L*d0
        x_flat=x_stacked.view(B,-1)
        if x_flat.shape[1]!=expected_flat_dim: raise RuntimeError(f"Flat shape error: Exp={expected_flat_dim}, Got={x_flat.shape[1]}")
        try: x_rechunked=x_flat.view(B,L_new,C_new)
        except RuntimeError as e: raise RuntimeError(f"Rechunk error: Flat={x_flat.shape}, Target=({B},{L_new},{C_new})") from e
        z_embedded=self.embed_layer_2(x_rechunked); z=self.embed_layer_2_act(z_embedded); return z

    def _positional_encoding(self, seq_len, embed_dim):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1); div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)); pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term); return pe

class LatentAttention_RL(nn.Module):
    def __init__(self, d_new: int, num_heads_latent: int, dropout: float, bias: bool):
        super().__init__(); assert d_new % num_heads_latent == 0; self.d_new = d_new; self.num_heads = num_heads_latent; self.head_dim = d_new // num_heads_latent; self.dropout_p = dropout; self.bias = bias
        self.c_attn = nn.Linear(d_new, 3 * d_new, bias=self.bias); self.c_proj = nn.Linear(d_new, d_new, bias=self.bias); self.attn_dropout = nn.Dropout(self.dropout_p); self.resid_dropout = nn.Dropout(self.dropout_p)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if self.flash: print(f"    - LatentAttention_RL: Using Flash Attn (d_new={d_new})")
        else: print(f"    - LatentAttention_RL: Using slow attn path.")

    def forward(self, z):
        B, L_new, C = z.size();
        if C != self.d_new: raise ValueError(f"LatentAttention C mismatch")
        q, k, v = self.c_attn(z).split(self.d_new, dim=2)
        q = q.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2); k = k.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2); v = v.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        if self.flash: y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0, is_causal=False)
        else: att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim)); att = F.softmax(att, dim=-1); att = self.attn_dropout(att); y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L_new, self.d_new); y = self.resid_dropout(self.c_proj(y)); return y

class LatentMLP_RL(nn.Module):
    def __init__(self, d_new: int, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__(); self.c_fc=nn.Linear(d_new, ff_latent_hidden, bias=bias); self.gelu=nn.GELU(); self.c_proj=nn.Linear(ff_latent_hidden, d_new, bias=bias); self.dropout=nn.Dropout(dropout)
    def forward(self, x): x=self.c_fc(x); x=self.gelu(x); x=self.c_proj(x); x=self.dropout(x); return x

class LMABlock_RL(nn.Module):
    def __init__(self, lma_config: LMAConfigRL, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__(); self.ln_1 = LayerNorm(lma_config.d_new, bias=bias); self.attn = LatentAttention_RL(lma_config.d_new, lma_config.num_heads_latent, dropout, bias); self.ln_2 = LayerNorm(lma_config.d_new, bias=bias); self.mlp = LatentMLP_RL(lma_config.d_new, ff_latent_hidden, dropout, bias)
    def forward(self, z): z = z + self.attn(self.ln_1(z)); z = z + self.mlp(self.ln_2(z)); return z

class LMAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space, embed_dim=64, num_heads_stacking=4, target_l_new=100, d_new=32, # Updated target_l_new default
        num_heads_latent=4, ff_latent_hidden=64, num_lma_layers=4, # Updated num_lma_layers default
        seq_len=200, dropout=0.1, bias=True # Updated seq_len default
    ):
        print("\n--- Initializing LMAFeaturesExtractor ---"); print("Calculating LMA dimensions...")
        # --- Use provided seq_len ---
        self.seq_len = seq_len
        # --------------------------
        self.lma_config = LMAConfigRL(seq_len=self.seq_len, embed_dim=embed_dim, num_heads_stacking=num_heads_stacking, target_l_new=target_l_new, d_new=d_new, num_heads_latent=num_heads_latent)
        print(f"  Final LMA Config: L={self.lma_config.seq_len}, d0={self.lma_config.embed_dim}, nh_stack={self.lma_config.num_heads_stacking}")
        print(f"                    L_new={self.lma_config.L_new}, C_new={self.lma_config.C_new}, d_new={self.lma_config.d_new}, nh_latent={self.lma_config.num_heads_latent}")
        feature_dim = self.lma_config.L_new * self.lma_config.d_new
        super().__init__(observation_space, features_dim=feature_dim)
        print(f"  SB3 features_dim (Flattened L_new * d_new): {feature_dim}")
        self.input_dim_total = observation_space.shape[0]
        if self.input_dim_total % self.seq_len != 0: raise ValueError(f"Input dimension ({self.input_dim_total}) must be divisible by seq_len ({self.seq_len}).")
        self.features_per_step = self.input_dim_total // self.seq_len
        self.initial_transform = LMA_InitialTransform_RL(features_per_step=self.features_per_step, lma_config=self.lma_config, dropout=dropout, bias=bias)
        self.lma_blocks = nn.ModuleList([ LMABlock_RL(lma_config=self.lma_config, ff_latent_hidden=ff_latent_hidden, dropout=dropout, bias=bias) for _ in range(num_lma_layers)])
        print(f"  Number of LMA Blocks: {num_lma_layers}"); self.flatten = nn.Flatten(); print("-----------------------------------------")

    def forward(self, x):
        batch_size = x.shape[0]
        try: x_reshaped = x.view(batch_size, self.seq_len, self.features_per_step)
        except RuntimeError as e: raise RuntimeError(f"Error reshaping input: Input={x.shape}, Target=({batch_size},{self.seq_len},{self.features_per_step})") from e
        z = self.initial_transform(x_reshaped)
        for block in self.lma_blocks: z = block(z)
        features = self.flatten(z); return features

class Transformer(BaseFeaturesExtractor):
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
        super(Transformer, self).__init__(observation_space, features_dim=feature_dim)

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
# Example Usage (Test LMA Extractor) - Updated for seq_len=200
#===============================================
if __name__ == "__main__":
    print("\nTesting classes.py definitions with LMA...")
    globals.COMPLEX_REWARD = True; globals.BASIC_REWARD = False; globals.RECTANGULAR_FOXTROT = False; globals.STATIONARY_FOXTROT = True; globals.RAND_POS = False; globals.FIXED_POS = True; globals.PROXIMITY_CCA = True; POSITIONAL_DEBUG = False; REWARD_DEBUG = False

    try:
        obs_hist_len_test = 200 # Use the new history length
        n_cca_test = 2         # Use the new number of CCAs
        obs_dim_per_step_test = (3 + 3) * n_cca_test + 3 # (pos+act)*num_cca + fox_pos
        total_obs_dim_test = obs_hist_len_test * obs_dim_per_step_test
        try:
            import gymnasium as gym
            dummy_obs_space = gym.spaces.Box(low=-grid_size, high=grid_size, shape=(total_obs_dim_test,), dtype=np.float32)
        except ImportError:
            class MockSpace: shape = (total_obs_dim_test,)
            dummy_obs_space = MockSpace()

        # Define LMA hyperparameters for testing with L=200
        lma_kwargs_test = dict(
            embed_dim=64,           # d0 - Keep relatively small for long sequence?
            num_heads_stacking=4,   # nh (64 % 4 == 0) -> dk=16
            target_l_new=100,       # Target L_new (L=200) -> find closest divisor for 200*64 = 12800
                                    # e.g., 100 is a divisor. L_new=100 -> C_new=128
            d_new=32,               # d_new - Latent dimension
            num_heads_latent=4,     # Latent heads (32 % 4 == 0) -> latent_dk=8
            ff_latent_hidden=64,    # Latent MLP hidden (2*d_new)
            num_lma_layers=4,       # More layers might be needed for longer sequence
            seq_len=obs_hist_len_test, # MUST be 200
            dropout=0.1,
            bias=True
        )

        lma_extractor = LMAFeaturesExtractor(
            observation_space=dummy_obs_space,
            **lma_kwargs_test
        )
        print("\nLMAFeaturesExtractor instantiated successfully for L=200.")

        batch_size = 4
        dummy_batch_obs = torch.randn(batch_size, total_obs_dim_test)
        features = lma_extractor(dummy_batch_obs)
        print(f"Input obs shape (batch): {dummy_batch_obs.shape}")
        print(f"Output features shape: {features.shape}")
        expected_feature_dim = lma_extractor.lma_config.L_new * lma_extractor.lma_config.d_new
        assert features.shape == (batch_size, expected_feature_dim), f"Feature dimension mismatch! Expected ({batch_size},{expected_feature_dim}), got {features.shape}"
        print("Forward pass successful.")

    except Exception as e:
        print(f"\nError testing LMAFeaturesExtractor: {e}"); import traceback; traceback.print_exc()
    print("\nclasses.py LMA testing complete.")