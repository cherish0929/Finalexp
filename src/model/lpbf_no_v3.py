"""
LPBF-NO v3 — Stage 1 Implementation
=====================================
Reference: LPBF_NO_v3_design.md

Architecture (Stage 1):
  SlotEncoder  →  N × LPBFMixerBlock  →  SlotIndexedDecoder

SlotEncoder:
  input = [γ(x), γ(Δx_laser), L(x,t), time_emb,
           slot_values [K], presence_mask [K], field_type_emb [K·d_type]]
  → MLP_shared → h_shared [N, enc_dim]
  → FiLM(laser_cond)
  → (if n_active > 1) per-slot Linear projection

LPBFMixerBlock (per active slot branch):
  DepthwiseConv3d  →  FactorizedCrossAttn  →  GenericCrossFieldAttn  →  KAN-FFN
  + block-level AttnRes + pre-norm + GatedResidual

SlotIndexedDecoder:
  TemperatureHead : ΔT = tanh(MLP(·)) × max_scale
  InterfaceHead   : φ = MLP(·) → α = σ(-φ/ε) clamped [0,1]
  GenericHead     : standard delta MLP

Coordinate convention (LPBF_NO_v3_design.md §0):
  x = laser scan direction (highest spatial freq → x_boost=2)
  y = depth direction (air above / metal below → Beer-Lambert decay along y)
  z = lateral scan-width direction

Grid storage order in dataset (dataset_fast.py line 66-70):
  for z: for y: for x:  →  flat index = z*Nx*Ny + y*Nx + x
  → einops rearrange: 'b (hz hy hx) c -> b c hx hy hz'

Stage 2 (not implemented here):
  Axial Bi-Mamba2 SSM (requires mamba_ssm library)
  Laser path context tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Dict, Tuple


# =============================================================================
# Positional / Fourier utilities  (mirrors physgto_attnres_max)
# =============================================================================

def fourier_embedding(pos: torch.Tensor, pos_start: int, pos_length: int) -> torch.Tensor:
    """Standard sinusoidal Fourier embedding."""
    orig_shape = pos.shape
    x = pos.reshape(-1, orig_shape[-1])
    idx = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq = 2 ** idx * math.pi
    cos_f = torch.cos(freq.view(1, 1, -1) * x.unsqueeze(-1))
    sin_f = torch.sin(freq.view(1, 1, -1) * x.unsqueeze(-1))
    emb = torch.cat([cos_f, sin_f], dim=-1).view(*orig_shape[:-1], -1)
    return torch.cat([emb, pos], dim=-1)


def fourier_embedding_pos(pos: torch.Tensor, pos_enc_dim: int, x_boost: int = 2) -> torch.Tensor:
    """
    Anisotropic Fourier position encoding (mirrors physgto_attnres_max).
    x-axis (scan direction) uses higher base frequency (+x_boost octaves).
    """
    orig_shape = pos.shape
    space_size = orig_shape[-1]
    x = pos.reshape(-1, space_size)
    idx = torch.arange(0, pos_enc_dim, device=pos.device).float()

    parts = []
    for dim_i in range(space_size):
        start = x_boost if dim_i == 0 else 0
        freq  = 2 ** (idx + start) * math.pi
        xi    = x[:, dim_i:dim_i + 1]
        parts.append(torch.cos(freq.unsqueeze(0) * xi))
        parts.append(torch.sin(freq.unsqueeze(0) * xi))

    emb = torch.cat(parts, dim=-1).view(*orig_shape[:-1], -1)
    return torch.cat([emb, pos], dim=-1)


# =============================================================================
# Analytic Laser Driving Field  L(x,t)
# =============================================================================

class LaserDrivingField(nn.Module):
    """
    Analytic laser driving field  L(x,t)  on the node grid.

    Gaussian heat source in the (x,z) scan plane with Beer-Lambert depth
    decay along y (depth axis).  When alpha.air field is not available the
    Beer-Lambert kernel degrades to a simple y < y_surface soft mask.

    Formula (LPBF_NO_v3_design.md §4.1):
        L(x,t) = (2·P·η / π·r²) · exp(-2·((x−xL)²+(z−zL)²) / r²) · K_y(y, α_air)
        K_y = exp(-κ · max(0, y_surface − y))   [depth below surface]

    Args:
        d_cond:     dimension of laser_cond FiLM vector (output of laser_mlp)
        d_film:     dimension of each FiLM (gamma, beta) = enc_dim
        kappa_init: initial absorption depth coefficient (learnable)
    """

    def __init__(self, cond_dim: int, enc_dim: int, kappa_init: float = 1e3):
        super().__init__()
        # Learnable kappa for Beer-Lambert depth kernel (scalar)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa_init)))

        # laser_cond: (P, r_norm, v_norm, scan_dir_4) → FiLM (γ, β)
        # We embed 4 laser scalars (indices 0-4 of thermal parameter)
        self.laser_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * enc_dim),  # outputs (gamma, beta) each enc_dim
        )
        nn.init.zeros_(self.laser_mlp[-1].weight)
        nn.init.zeros_(self.laser_mlp[-1].bias)

    def forward(
        self,
        node_pos_physical: torch.Tensor,   # [B, N, 3]  un-normalized (x, y, z)
        conditions: torch.Tensor,           # [B, cond_dim]
        spatial_inform: torch.Tensor,       # [B, 10]
        alpha_air: Optional[torch.Tensor] = None,  # [B, N, 1] or None
    ):
        """
        Returns:
            L_field:    [B, N, 1]   energy density field
            laser_cond: [B, enc_dim] FiLM gamma
            laser_beta: [B, enc_dim] FiLM beta
            delta_x_laser: [B, N, 3] relative displacement x - x_laser(t)
        """
        B, N, _ = node_pos_physical.shape
        device   = node_pos_physical.device

        # ---- Laser position estimate from spatial_inform ----
        # spatial_inform [10]: bounds[0:6], ds_shape[6:9], time_ref[9]
        # Domain bounds: [xmin, xmax, ymin, ymax, zmin, zmax] (indices 0-5)
        # Use domain centre as a prior for laser x,z; y-surface from ymin
        bounds = spatial_inform[:, :6]       # [B, 6]
        x_centre = (bounds[:, 0] + bounds[:, 1]) * 0.5  # [B]
        z_centre = (bounds[:, 4] + bounds[:, 5]) * 0.5  # [B]
        y_surface = bounds[:, 2]                         # [B]  ymin ≈ top surface

        # ---- Extract laser scalars from conditions ----
        # _process_condition_normalize encodes 5 thermal params at indices 0-4
        # thermal[3]=P_norm, thermal[4]=r_norm, thermal[5]=v_norm,
        # thermal[7]=absorb1_norm, thermal[8]=absorb2_norm
        # conditions is a concatenation; first 5 dims are the thermal params
        laser_scalars = conditions[:, :4].float()   # [B, 4]

        # ---- FiLM conditioning ----
        film_out = self.laser_mlp(laser_scalars)     # [B, 2*enc_dim]
        laser_gamma, laser_beta = film_out.chunk(2, dim=-1)

        # ---- Reconstruct approximate physical laser position ----
        # (In full implementation, x_laser(t) would come from condition/data;
        #  here we use domain centre as a default prior)
        x_laser = x_centre.view(B, 1, 1).expand(B, 1, 3)   # [B, 1, 3]
        x_laser = torch.cat([
            x_centre.view(B, 1, 1),
            y_surface.view(B, 1, 1),
            z_centre.view(B, 1, 1),
        ], dim=-1)  # [B, 1, 3]

        # ---- Relative displacement Δx = x - x_laser(t) ----
        delta_x = node_pos_physical - x_laser   # [B, N, 3]

        # ---- Gaussian in (x,z) plane ----
        # Use a fixed reference radius (physical) scaled by a learnable factor
        # In full v3 this would use the actual beam radius from conditions
        r_ref = 5e-5   # 50 µm reference radius (typical LPBF)
        dx = delta_x[..., 0]  # x-displacement
        dz = delta_x[..., 2]  # z-displacement
        gauss = torch.exp(-2.0 * (dx ** 2 + dz ** 2) / (r_ref ** 2 + 1e-20))  # [B, N]

        # ---- Beer-Lambert depth kernel along y ----
        kappa = torch.exp(self.log_kappa)
        y_node = node_pos_physical[..., 1]           # [B, N]
        y_surf = y_surface.unsqueeze(-1)             # [B, 1]
        depth_below = torch.clamp(y_surf - y_node, min=0.0)  # [B, N]

        if alpha_air is not None:
            # Smooth surface indicator: only absorb where alpha.air → 0 (metal)
            metal_frac = 1.0 - alpha_air.squeeze(-1).clamp(0, 1)   # [B, N]
            depth_kernel = torch.exp(-kappa * depth_below) * metal_frac
        else:
            # Fallback: purely geometric depth kernel
            depth_kernel = torch.exp(-kappa * depth_below)

        # ---- Assemble L(x,t) ----
        L_field = (gauss * depth_kernel).unsqueeze(-1)   # [B, N, 1]

        return L_field, laser_gamma, laser_beta, delta_x


# =============================================================================
# Basic building blocks
# =============================================================================

class MLP(nn.Module):
    """Simple MLP with optional LayerNorm."""
    def __init__(self, dims: List[int], act: str = "silu", layer_norm: bool = True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU() if act == "silu" else nn.GELU())
        if layer_norm:
            layers.append(nn.LayerNorm(dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GatedResidual(nn.Module):
    """
    Gated residual: out = x + layer_scale * sigmoid(gate(x)) * delta
    Initializes to near-identity (layer_scale_init=1e-2, gate bias=0).
    """
    def __init__(self, dim: int, layer_scale_init: float = 1e-2):
        super().__init__()
        self.layer_scale = nn.Parameter(torch.tensor(layer_scale_init))
        self.gate_proj   = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_proj(x))
        return x + self.layer_scale * g * delta


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


def block_attn_res(
    history: List[torch.Tensor],
    current: torch.Tensor,
    w: nn.Parameter,
    rms_norm: RMSNorm,
) -> torch.Tensor:
    """Softmax attention residual over block history + current."""
    sources = history + [current]
    V = torch.stack(sources, dim=0)              # [S, B, N, D]
    K = rms_norm(V)
    logits = torch.einsum("d, s b n d -> s b n", w, K).clamp(-30, 30)
    alpha  = logits.softmax(dim=0)
    return torch.einsum("s b n, s b n d -> b n d", alpha, V)


# =============================================================================
# Efficient KAN Layer (B-spline based, Efficient-KAN style)
# =============================================================================

class EfficientKANLayer(nn.Module):
    """
    Simplified efficient KAN layer using learnable B-spline activations.

    Reference: Blealtan, "An Efficient Implementation of KAN", 2024.
    Each input dimension gets an independent B-spline activation with
    `grid_size` basis functions and `spline_order` degree.

    This implementation uses a grid of Chebyshev-spaced knots and stores
    the spline coefficients as learnable parameters. The base linear
    transformation is kept alongside for capacity.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
    ):
        super().__init__()
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.grid_size   = grid_size
        self.spline_order = spline_order

        # Base linear weight (residual path)
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # Spline coefficients: shape [out_dim, in_dim, grid_size + spline_order]
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.empty(out_dim, in_dim, n_basis)
        )
        nn.init.normal_(self.spline_weight, std=scale_noise / math.sqrt(in_dim))

        # Grid points (not learnable): Chebyshev-spaced in [-1, 1]
        # extended with order extra points on each side
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - spline_order * h, 1 + spline_order * h,
                               grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)

        self.act = nn.SiLU()

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions.
        x: [..., in_dim]
        Returns: [..., in_dim, n_basis]
        """
        x = x.unsqueeze(-1)  # [..., in_dim, 1]
        grid = self.grid      # [n_knots]
        # Clamp to grid range
        x = x.clamp(self.grid[0], self.grid[-1])

        # Order 0 basis
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # [..., in_dim, n_knots-1]

        # Recurrence for higher orders
        for k in range(1, self.spline_order + 1):
            left  = (x - grid[:-(k + 1)]) / (grid[k:-1]   - grid[:-(k + 1)] + 1e-8)
            right = (grid[(k + 1):] - x)  / (grid[(k + 1):] - grid[1:-k]    + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]

        return bases  # [..., in_dim, n_basis]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_dim]
        returns: [..., out_dim]
        """
        orig = x.shape
        x_flat = x.reshape(-1, self.in_dim)     # [M, in_dim]

        # Base (linear residual with SiLU)
        base_out = F.linear(self.act(x_flat), self.base_weight)  # [M, out_dim]

        # Spline contribution
        splines = self._b_splines(x_flat)               # [M, in_dim, n_basis]
        # spline_weight: [out_dim, in_dim, n_basis]
        spline_out = torch.einsum(
            "m i k, o i k -> m o", splines, self.spline_weight
        )  # [M, out_dim]

        out = base_out + spline_out
        return out.reshape(*orig[:-1], self.out_dim)


# =============================================================================
# Physics-KAN-FFN
# =============================================================================

class PhysicsKANFFN(nn.Module):
    """
    Physics-guided KAN-FFN (LPBF_NO_v3_design.md §4.6).

    Architecture:
      Main branch: standard 4×dim FFN (primary capacity)
      KAN branches: n_src lightweight EfficientKANLayer branches
                    (laser_heating, phase_change, surface_tension, buoyancy)
      Gate: softmax-weighted combination conditioned on (h, L_field)

    The gate weights' spatial distribution is directly interpretable as
    "source term dominance maps" — a key paper-level visualization.
    """

    def __init__(
        self,
        dim: int,
        n_src: int = 4,
        d_src: Optional[int] = None,
        grid_size: int = 5,
        spline_order: int = 3,
        layer_scale_init: float = 1e-2,
    ):
        super().__init__()
        d_src = d_src or max(16, dim // n_src)
        self.n_src = n_src
        self.d_src = d_src
        self.dim   = dim

        # Main FFN (4× expansion)
        self.main = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

        # KAN branches (source terms: laser, phase-change, surface-tension, buoyancy)
        self.kan_branches = nn.ModuleList([
            EfficientKANLayer(dim, d_src, grid_size=grid_size, spline_order=spline_order)
            for _ in range(n_src)
        ])

        # Project concatenated KAN outputs back to dim
        self.kan_out_proj = nn.Linear(n_src * d_src, dim)

        # Gate: depends on (h, L_field) → n_src weights
        # L_field is a single intensity scalar per node
        self.gate = nn.Sequential(
            nn.LayerNorm(dim + 1),
            nn.Linear(dim + 1, n_src),
        )

        # Gated residual for final fusion
        self.gated_res = GatedResidual(dim, layer_scale_init)

    def forward(self, h: torch.Tensor, L_field: torch.Tensor) -> torch.Tensor:
        """
        h:       [B, N, dim]
        L_field: [B, N, 1]  laser energy density
        Returns: [B, N, dim]
        """
        # Main branch
        h_main = self.main(h)   # [B, N, dim]

        # KAN branches
        kan_outs = [branch(h) for branch in self.kan_branches]   # n_src × [B, N, d_src]

        # Softmax gate conditioned on (h, L)
        gate_in  = torch.cat([h, L_field], dim=-1)               # [B, N, dim+1]
        gate_w   = torch.softmax(self.gate(gate_in), dim=-1)     # [B, N, n_src]

        # Weighted sum over KAN branches then project
        kan_cat  = torch.cat(kan_outs, dim=-1)                   # [B, N, n_src*d_src]
        # Weight each branch's output by gate
        gate_exp = gate_w.repeat_interleave(self.d_src, dim=-1)  # [B, N, n_src*d_src]
        kan_phys = self.kan_out_proj(kan_cat * gate_exp)          # [B, N, dim]

        physics_out = h_main + kan_phys
        return self.gated_res(h, physics_out)


# =============================================================================
# 3D Depth-wise Convolution (local feature extraction, replaces GNN)
# =============================================================================

class DepthwiseConv3d(nn.Module):
    """
    3D depth-wise separable convolution for local feature extraction
    on regular grids (LPBF_NO_v3_design.md §4.3).

    Replaces GNN message passing on the regular voxel grid.
    Equivalent to 6-neighbor aggregation but exploits cuDNN 3D conv
    efficiency — no scatter/gather overhead.

    Input/output format: flat node features [B, N, C]
    Grid rearrangement uses z-y-x storage order from dataset_fast.py:
        rearrange 'b (hz hy hx) c -> b c hx hy hz'

    Args:
        dim:       channel dimension
        grid_shape: (Hx, Hy, Hz) — must match when forward is called
        kernel_size: kernel for each axis (default 3, isotropic)
        layer_scale_init: for GatedResidual
    """

    def __init__(self, dim: int, layer_scale_init: float = 1e-2):
        super().__init__()
        # Depth-wise conv (each channel independently)
        self.dw_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        # Point-wise mixing (1×1×1)
        self.pw_conv = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.norm    = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.act     = nn.GELU()
        self.res     = GatedResidual(dim, layer_scale_init)

        nn.init.kaiming_normal_(self.dw_conv.weight)
        nn.init.ones_(self.pw_conv.weight)

    def forward(
        self,
        h: torch.Tensor,        # [B, N, C]
        grid_shape: Tuple[int, int, int],  # (Hx, Hy, Hz)
    ) -> torch.Tensor:
        Hx, Hy, Hz = grid_shape
        B, N, C = h.shape
        assert N == Hx * Hy * Hz, f"Node count {N} ≠ {Hx}×{Hy}×{Hz}={Hx*Hy*Hz}"

        # Rearrange: z-y-x storage → c h_x h_y h_z layout
        h_3d = rearrange(h, "b (hz hy hx) c -> b c hx hy hz",
                         hx=Hx, hy=Hy, hz=Hz)  # [B, C, Hx, Hy, Hz]

        h_loc = self.dw_conv(h_3d)              # depth-wise
        h_loc = self.pw_conv(self.act(self.norm(h_loc)))  # point-wise

        h_flat = rearrange(h_loc, "b c hx hy hz -> b (hz hy hx) c")
        return self.res(h, h_flat)


# =============================================================================
# Factorized Cross-Attention (global information, O(M·N))
# =============================================================================

class FactorizedCrossAttn(nn.Module):
    """
    Factorized cross-attention for global information aggregation
    (LPBF_NO_v3_design.md §4.4 / FactFormer-style).

    Learnable M global queries cross-attend to x-, y-, z-axis slices
    independently, their outputs are fused, then h cross-attends back
    to the fused global tokens.

    Complexity: O(M · (Hx + Hy + Hz)) per batch, independent of N.

    Args:
        dim:       feature dimension
        n_heads:   attention heads
        n_tokens:  number of global learnable query tokens M
        layer_scale_init: for GatedResidual
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_tokens: int = 64,
        layer_scale_init: float = 1e-2,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        self.global_tokens = nn.Parameter(torch.empty(n_tokens, dim))
        nn.init.xavier_uniform_(self.global_tokens)

        # Cross-attention: queries → each axis
        self.cross_attn_x = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
        self.cross_attn_y = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
        self.cross_attn_z = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)

        # Self-attention to fuse the 3 axis-wise global token sets
        self.token_self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
        self.token_norm      = nn.LayerNorm(dim)

        # Write-back: h cross-attends to refined tokens
        self.writeback_norm   = nn.LayerNorm(dim)
        self.writeback_attn   = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
        self.res              = GatedResidual(dim, layer_scale_init)

    def forward(
        self,
        h: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        h: [B, N, dim], grid_shape: (Hx, Hy, Hz)
        """
        Hx, Hy, Hz = grid_shape
        B = h.shape[0]

        # Rearrange to 3D: z-y-x storage → 3D grid
        h_3d = rearrange(h, "b (hz hy hx) c -> b hx hy hz c",
                         hx=Hx, hy=Hy, hz=Hz)  # [B, Hx, Hy, Hz, C]

        Q = self.global_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, M, C]

        # ---- Cross-attend to each axis ----
        # x-axis: treat (Hy*Hz) batch dimension, seq_len = Hx
        kv_x = rearrange(h_3d, "b hx hy hz c -> (b hy hz) hx c")
        Q_x  = Q.unsqueeze(1).expand(-1, Hy * Hz, -1, -1).reshape(B * Hy * Hz, self.n_tokens, -1)
        tok_x, _ = self.cross_attn_x(Q_x, kv_x, kv_x)   # [(B*Hy*Hz), M, C]
        tok_x = tok_x.reshape(B, Hy * Hz, self.n_tokens, -1).mean(dim=1)  # [B, M, C]

        # y-axis
        kv_y = rearrange(h_3d, "b hx hy hz c -> (b hx hz) hy c")
        Q_y  = Q.unsqueeze(1).expand(-1, Hx * Hz, -1, -1).reshape(B * Hx * Hz, self.n_tokens, -1)
        tok_y, _ = self.cross_attn_y(Q_y, kv_y, kv_y)
        tok_y = tok_y.reshape(B, Hx * Hz, self.n_tokens, -1).mean(dim=1)  # [B, M, C]

        # z-axis
        kv_z = rearrange(h_3d, "b hx hy hz c -> (b hx hy) hz c")
        Q_z  = Q.unsqueeze(1).expand(-1, Hx * Hy, -1, -1).reshape(B * Hx * Hy, self.n_tokens, -1)
        tok_z, _ = self.cross_attn_z(Q_z, kv_z, kv_z)
        tok_z = tok_z.reshape(B, Hx * Hy, self.n_tokens, -1).mean(dim=1)  # [B, M, C]

        # ---- Fuse: sum + self-attention ----
        tokens = tok_x + tok_y + tok_z                      # [B, M, C]
        tokens_in = self.token_norm(tokens)
        tokens, _ = self.token_self_attn(tokens_in, tokens_in, tokens_in)

        # ---- Write back: h → tokens ----
        h_normed   = self.writeback_norm(h)
        glob_out, _ = self.writeback_attn(h_normed, tokens, tokens)  # [B, N, C]
        return self.res(h, glob_out)


# =============================================================================
# Universal Cross-Field Gate (LPBF_NO_v3_design.md §3)
# =============================================================================

class UniversalCrossFieldGate(nn.Module):
    """
    Three-tier degradable gate for cross-field attention residual weighting.

    Tiers (selected at runtime by n_interface_active):
      0 interface  →  Laser-Only Gate:      g = σ(MLP(L))
      1 interface  →  Single-Interface:     g = σ(MLP(L, ‖∇α‖))
      2 interfaces →  Dual-Interface:       g = σ(MLP(L, ‖∇α‖, ‖∇γ‖, ‖∇α·∇γ‖))

    All tiers return [B, N, 1].
    """

    def __init__(self, d_gate: int = 32):
        super().__init__()
        self.gate_laser  = nn.Sequential(nn.Linear(1, d_gate), nn.SiLU(), nn.Linear(d_gate, 1))
        self.gate_single = nn.Sequential(nn.Linear(2, d_gate), nn.SiLU(), nn.Linear(d_gate, 1))
        self.gate_dual   = nn.Sequential(nn.Linear(4, d_gate), nn.SiLU(), nn.Linear(d_gate, 1))

    def _grad_magnitude(self, h: torch.Tensor, grid_shape: Tuple) -> torch.Tensor:
        """Approximate spatial gradient magnitude using 3D FD on channel mean."""
        Hx, Hy, Hz = grid_shape
        B, N, C = h.shape
        # Use mean over channels as a scalar field proxy
        h_scalar = h.mean(dim=-1, keepdim=True)                       # [B, N, 1]
        h_3d = rearrange(h_scalar, "b (hz hy hx) c -> b c hx hy hz",
                         hx=Hx, hy=Hy, hz=Hz)
        # Central difference along each axis (pad to keep size)
        gx = F.pad(h_3d[:, :, 1:] - h_3d[:, :, :-1], (0, 0, 0, 0, 0, 1))
        gy = F.pad(h_3d[:, :, :, 1:] - h_3d[:, :, :, :-1], (0, 0, 0, 1))
        gz = F.pad(h_3d[:, :, :, :, 1:] - h_3d[:, :, :, :, :-1], (0, 1))
        mag = (gx ** 2 + gy ** 2 + gz ** 2).sqrt()
        return rearrange(mag, "b c hx hy hz -> b (hz hy hx) c")       # [B, N, 1]

    def forward(
        self,
        L_field: torch.Tensor,                  # [B, N, 1]
        interface_branches: List[torch.Tensor], # list of [B, N, enc_dim], len 0-2
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Returns gate weights [B, N, 1] in (0, 1)."""
        n = len(interface_branches)

        if n == 0:
            return torch.sigmoid(self.gate_laser(L_field))

        if n == 1:
            grad_mag = self._grad_magnitude(interface_branches[0], grid_shape)
            inp = torch.cat([L_field, grad_mag], dim=-1)
            return torch.sigmoid(self.gate_single(inp))

        # n >= 2 (use first two)
        g1 = self._grad_magnitude(interface_branches[0], grid_shape)
        g2 = self._grad_magnitude(interface_branches[1], grid_shape)
        # Triple-phase line indicator: inner product of gradient vectors
        Hx, Hy, Hz = grid_shape
        B, N, C = interface_branches[0].shape

        def grad_vec(h):
            h_3d = rearrange(h.mean(-1, keepdim=True),
                             "b (hz hy hx) c -> b c hx hy hz", hx=Hx, hy=Hy, hz=Hz)
            gx = F.pad(h_3d[:, :, 1:] - h_3d[:, :, :-1], (0, 0, 0, 0, 0, 1))
            gy = F.pad(h_3d[:, :, :, 1:] - h_3d[:, :, :, :-1], (0, 0, 0, 1))
            gz = F.pad(h_3d[:, :, :, :, 1:] - h_3d[:, :, :, :, :-1], (0, 1))
            vx = rearrange(gx, "b c hx hy hz -> b (hz hy hx) c")
            vy = rearrange(gy, "b c hx hy hz -> b (hz hy hx) c")
            vz = rearrange(gz, "b c hx hy hz -> b (hz hy hx) c")
            return torch.cat([vx, vy, vz], dim=-1)  # [B, N, 3]

        v1 = grad_vec(interface_branches[0])
        v2 = grad_vec(interface_branches[1])
        dot = (v1 * v2).sum(-1, keepdim=True).abs()   # [B, N, 1]
        inp = torch.cat([L_field, g1, g2, dot], dim=-1)
        return torch.sigmoid(self.gate_dual(inp))


# =============================================================================
# Generic Cross-Field Attention (multi-slot interaction)
# =============================================================================

class GenericCrossFieldAttn(nn.Module):
    """
    Multi-slot cross-field attention.
    Reuses the BiCondFieldCrossAttention pattern from physgto_attnres_max
    but simplified to a single latent refinement layer for Stage 1.

    For each active slot i, other slots are compressed into M tokens,
    refined, and written back via gated residual scaled by UniversalGate.

    Skipped entirely when n_active == 1.
    """

    def __init__(
        self,
        enc_dim: int,
        n_heads: int = 4,
        n_token: int = 32,
        layer_scale_init: float = 1e-2,
    ):
        super().__init__()
        self.n_token = n_token

        # Learnable base queries
        self.Q_base = nn.Parameter(torch.empty(n_token, enc_dim))
        nn.init.xavier_uniform_(self.Q_base)

        # Query offsets: self_summary, other_summary, pair_summary
        self.q_self_proj  = nn.Linear(enc_dim, enc_dim)
        self.q_other_proj = nn.Linear(enc_dim, enc_dim)
        self.q_pair_proj  = nn.Linear(enc_dim, enc_dim)
        self.q_norm       = nn.LayerNorm(enc_dim)

        # Self-aware gate for other-field features
        self.other_gate = nn.Linear(enc_dim, enc_dim)

        # Cross-attention (Q from self, KV from gated_other)
        self.cross_attn = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True, dropout=0.0)

        # Single latent refinement
        self.lat_norm  = nn.LayerNorm(enc_dim)
        self.lat_attn  = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True, dropout=0.0)
        self.lat_ffn   = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, 2 * enc_dim),
            nn.SiLU(),
            nn.Linear(2 * enc_dim, enc_dim),
        )
        self.lat_res1 = GatedResidual(enc_dim, layer_scale_init)
        self.lat_res2 = GatedResidual(enc_dim, layer_scale_init)

        # Write-back
        self.out_norm  = nn.LayerNorm(enc_dim)
        self.out_attn  = nn.MultiheadAttention(enc_dim, n_heads, batch_first=True, dropout=0.0)
        self.out_res   = GatedResidual(enc_dim, layer_scale_init)

    def forward(
        self,
        h_self: torch.Tensor,              # [B, N, C]
        h_others: List[torch.Tensor],      # list of [B, N, C]
    ) -> torch.Tensor:
        B = h_self.shape[0]

        # ---- Build dynamic query ----
        self_s  = h_self.mean(1, keepdim=True)             # [B, 1, C]
        # Aggregate other fields summary
        others_mean = torch.stack(h_others, dim=0).mean(0).mean(1, keepdim=True)  # [B, 1, C]
        # Self-aware gated other
        gated_other0 = h_others[0] * torch.sigmoid(self.other_gate(h_self))
        pair_s  = (h_self * gated_other0).mean(1, keepdim=True)  # [B, 1, C]

        Q = self.Q_base.unsqueeze(0).expand(B, -1, -1)
        Q = Q + self.q_self_proj(self_s)
        Q = Q + self.q_other_proj(others_mean)
        Q = Q + self.q_pair_proj(pair_s)
        Q = self.q_norm(Q)                                 # [B, M, C]

        # ---- Cross-attend to gated others (concat for multi-field) ----
        gated_others = [
            h_o * torch.sigmoid(self.other_gate(h_self))
            for h_o in h_others
        ]
        kv = torch.cat(gated_others, dim=1)               # [B, n_others*N, C]
        tokens, _ = self.cross_attn(Q, kv, kv)             # [B, M, C]

        # ---- Latent refinement ----
        t_in   = self.lat_norm(tokens)
        t_attn, _ = self.lat_attn(t_in, t_in, t_in)
        tokens = self.lat_res1(tokens, t_attn)
        tokens = self.lat_res2(tokens, self.lat_ffn(tokens))

        # ---- Write back ----
        h_normed   = self.out_norm(h_self)
        out, _     = self.out_attn(h_normed, tokens, tokens)
        return self.out_res(h_self, out)


# =============================================================================
# Slot-based Shared Encoder
# =============================================================================

class SlotEncoder(nn.Module):
    """
    Fixed-slot shared encoder (LPBF_NO_v3_design.md §2.3).

    input_dim = (
        (1 + 2*pos_enc_dim) * space_size       # γ(x): Fourier + raw coords
      + (1 + 2*pos_enc_dim) * space_size       # γ(Δx_laser): relative laser coords
      + 1                                       # L(x,t): laser energy density
      + (1 + 2*pos_enc_dim) * 2 + (1+2*2)      # time_emb (relative + low-freq)
      + K                                       # slot_values
      + K                                       # presence_mask
      + K * d_type                              # field_type_embedding
    )

    The dimension is FIXED regardless of how many fields are active.
    """

    def __init__(
        self,
        K: int,
        enc_dim: int,
        space_size: int = 3,
        pos_enc_dim: int = 6,
        x_boost: int = 2,
        d_type: int = 16,
        n_type: int = 4,
    ):
        super().__init__()
        self.K          = K
        self.enc_dim    = enc_dim
        self.pos_enc_dim = pos_enc_dim
        self.x_boost    = x_boost
        self.d_type     = d_type

        # Per-slot type embedding (K learnable vectors of dim d_type)
        self.type_emb = nn.Embedding(n_type, d_type)

        # Fourier position dim: raw + 2*pos_enc_dim per axis
        pos_feat_dim = space_size + 2 * pos_enc_dim * space_size

        # Time encoding dim (mirrors physgto_attnres_max._encode_time):
        # FourierEmbedding(time+dt, 0, pos_enc_dim) → (1+2*pos_enc_dim)*2
        # FourierEmbedding(time, 0, 2) → (1+2*2) = 5
        enc_t_dim = (1 + 2 * pos_enc_dim) * 2 + (1 + 2 * 2)

        input_dim = (
            pos_feat_dim     # γ(x)
            + pos_feat_dim   # γ(Δx_laser)
            + 1              # L(x,t)
            + enc_t_dim      # time
            + K              # slot values
            + K              # presence mask
            + K * d_type     # type embeddings (flattened)
        )

        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, enc_dim * 2),
            nn.SiLU(),
            nn.Linear(enc_dim * 2, enc_dim),
            nn.LayerNorm(enc_dim),
        )

    def forward(
        self,
        node_pos: torch.Tensor,          # [B, N, 3] normalized [0,1]
        delta_x_laser: torch.Tensor,     # [B, N, 3] physical, relative to laser
        L_field: torch.Tensor,           # [B, N, 1]
        time_emb: torch.Tensor,          # [B, N, enc_t_dim]
        slot_values: torch.Tensor,       # [B, N, K]
        presence_mask: torch.Tensor,     # [B, K]
        slot_type_indices: torch.Tensor, # [K] — shared across batch
        laser_gamma: torch.Tensor,       # [B, enc_dim]
        laser_beta: torch.Tensor,        # [B, enc_dim]
    ) -> torch.Tensor:
        """Returns h_shared [B, N, enc_dim]."""
        B, N, _ = node_pos.shape

        # Fourier position encodings
        pos_enc = fourier_embedding_pos(node_pos, self.pos_enc_dim, self.x_boost)   # [B, N, pos_feat]

        # Normalize delta_x_laser for Fourier (scale to approx unit range)
        delta_x_norm = delta_x_laser / (1e-4 + 1e-8)  # typical laser offset ~100µm
        laser_pos_enc = fourier_embedding_pos(delta_x_norm, self.pos_enc_dim, self.x_boost)

        # Presence mask broadcast: [B, N, K]
        pm = presence_mask.unsqueeze(1).expand(B, N, self.K)

        # Type embedding: [K, d_type] → broadcast to [B, N, K*d_type]
        type_e = self.type_emb(slot_type_indices)                  # [K, d_type]
        type_e = type_e.view(1, 1, self.K * self.d_type).expand(B, N, -1)

        feats = torch.cat([
            pos_enc,
            laser_pos_enc,
            L_field,
            time_emb,
            slot_values,
            pm,
            type_e,
        ], dim=-1)   # [B, N, input_dim]

        h = self.shared_mlp(feats)

        # FiLM modulation by laser conditioning
        h = laser_gamma.unsqueeze(1) * h + laser_beta.unsqueeze(1)
        return h  # [B, N, enc_dim]


# =============================================================================
# LPBF Mixer Block (Stage 1)
# =============================================================================

class LPBFMixerBlock(nn.Module):
    """
    Single LPBF-Mixer block (Stage 1, LPBF_NO_v3_design.md §4):
        DepthwiseConv3d
        → FactorizedCrossAttn
        → GenericCrossFieldAttn  (skipped if n_active == 1)
        → PhysicsKANFFN
        + Block-level AttnRes at every sublayer

    Per-slot branches share the DepthConv / FactAttn modules (one instance
    per slot); GenericCrossFieldAttn is shared across slots.
    """

    # Sublayer index constants
    IDX_CONV  = 0
    IDX_FACT  = 1
    IDX_CROSS = 2
    IDX_FFN   = 3
    N_SUB     = 4

    def __init__(
        self,
        K: int,
        enc_dim: int,
        n_heads: int = 4,
        n_tokens: int = 64,
        n_cross_tokens: int = 32,
        n_src: int = 4,
        d_src: Optional[int] = None,
        layer_scale_init: float = 1e-2,
        d_gate: int = 32,
    ):
        super().__init__()
        self.K       = K
        self.enc_dim = enc_dim

        # --- Per-slot modules ---
        self.depth_convs = nn.ModuleList([
            DepthwiseConv3d(enc_dim, layer_scale_init) for _ in range(K)
        ])
        self.fact_attns = nn.ModuleList([
            FactorizedCrossAttn(enc_dim, n_heads, n_tokens, layer_scale_init)
            for _ in range(K)
        ])
        self.kan_ffns = nn.ModuleList([
            PhysicsKANFFN(enc_dim, n_src, d_src, layer_scale_init=layer_scale_init)
            for _ in range(K)
        ])

        # --- Cross-field module (single shared instance) ---
        self.cross_field_attn = GenericCrossFieldAttn(
            enc_dim, n_heads, n_cross_tokens, layer_scale_init
        )
        self.universal_gate = UniversalCrossFieldGate(d_gate)
        # Layer scale for gated cross residual
        self.cross_layer_scale = nn.Parameter(torch.tensor(layer_scale_init))

        # --- Block-level AttnRes (K slots × N_SUB sublayers) ---
        self.block_attn_res_w = nn.ParameterList([
            nn.Parameter(torch.zeros(enc_dim))
            for _ in range(K * self.N_SUB)
        ])
        self.rms_norm = RMSNorm()

    def _bw(self, slot_idx: int, sub_idx: int) -> nn.Parameter:
        return self.block_attn_res_w[slot_idx * self.N_SUB + sub_idx]

    def _apply_bar(self, history, current, slot_idx, sub_idx):
        return block_attn_res(history, current, self._bw(slot_idx, sub_idx), self.rms_norm)

    def forward(
        self,
        h_branches: Dict[int, torch.Tensor],   # {slot_idx: [B, N, enc_dim]}
        L_field: torch.Tensor,                 # [B, N, 1]
        grid_shape: Tuple[int, int, int],
        blocks_history: Dict[int, List[torch.Tensor]],  # per-slot block history
        slot_type_strs: List[str],             # len K, type string per slot
        active_indices: List[int],             # currently active slots
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[torch.Tensor]]]:
        """
        Returns:
            h_branches:    updated {slot_idx: [B, N, enc_dim]}
            blocks_history: updated history (current block appended)
        """
        h = {k: v for k, v in h_branches.items()}  # shallow copy

        # =====================================================================
        # Sublayer 0: 3D DepthConv
        # =====================================================================
        for i in active_indices:
            v_h = self._apply_bar(blocks_history[i], h[i], i, self.IDX_CONV)
            h[i] = self.depth_convs[i](v_h, grid_shape)

        # =====================================================================
        # Sublayer 1: Factorized Cross-Attn
        # =====================================================================
        for i in active_indices:
            v_h = self._apply_bar(blocks_history[i], h[i], i, self.IDX_FACT)
            h[i] = self.fact_attns[i](v_h, grid_shape)

        # =====================================================================
        # Sublayer 2: Generic Cross-Field Attention (skip if n_active == 1)
        # =====================================================================
        if len(active_indices) > 1:
            # Identify interface slots for the universal gate
            interface_branches = [
                h[i] for i in active_indices
                if slot_type_strs[i] == "interface"
            ]

            h_after_cross: Dict[int, torch.Tensor] = {}
            for i in active_indices:
                v_h   = self._apply_bar(blocks_history[i], h[i], i, self.IDX_CROSS)
                others = [h[j] for j in active_indices if j != i]
                cross_out = self.cross_field_attn(v_h, others)

                # Universal gate modulates cross residual
                gate = self.universal_gate(L_field, interface_branches, grid_shape)
                h_after_cross[i] = h[i] + self.cross_layer_scale * gate * (cross_out - h[i])

            h = h_after_cross

        # =====================================================================
        # Sublayer 3: Physics-KAN-FFN
        # =====================================================================
        for i in active_indices:
            v_h  = self._apply_bar(blocks_history[i], h[i], i, self.IDX_FFN)
            h[i] = self.kan_ffns[i](v_h, L_field)

        # Update block history
        for i in active_indices:
            blocks_history[i] = blocks_history[i] + [h[i]]

        return h, blocks_history


# =============================================================================
# Slot-Indexed Decoder
# =============================================================================

class TemperatureHead(nn.Module):
    """
    Bounded residual prediction for temperature (LPBF_NO_v3_design.md §4.7).
    ΔT = tanh(MLP(h_agg)) × max_delta_scale
    """

    def __init__(self, enc_dim: int, n_block: int, pos_enc_dim: int, max_delta: float = 500.0):
        super().__init__()
        self.max_delta = max_delta
        enc_s_dim = 3 + 2 * pos_enc_dim * 3  # matches fourier_embedding_pos output

        self.attn_net = nn.Sequential(
            nn.Linear(enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, n_block),
        )
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(enc_dim + enc_s_dim),
            nn.Linear(enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, 1),
        )

    def forward(
        self,
        V_all: torch.Tensor,   # [B, n_block, N, enc_dim]
        pos_enc: torch.Tensor, # [B, N, enc_s_dim]
    ) -> torch.Tensor:         # [B, N, 1]
        w = torch.softmax(self.attn_net(pos_enc).clamp(-30, 30), dim=-1)  # [B, N, n_block]
        V_perm = V_all.permute(0, 2, 1, 3)                                 # [B, N, n_block, C]
        V_agg  = (w.unsqueeze(-1) * V_perm).sum(2)                         # [B, N, C]
        delta  = self.delta_mlp(torch.cat([V_agg, pos_enc], dim=-1))       # [B, N, 1]
        return torch.tanh(delta) * self.max_delta


class InterfaceHead(nn.Module):
    """
    Signed-distance proxy head for VoF interface fields
    (LPBF_NO_v3_design.md §4.7).
    φ = MLP(h_agg) → α = σ(-φ/ε), clamped to [0,1].
    ε is annealed during training (passed externally).
    """

    def __init__(
        self,
        enc_dim: int,
        n_block: int,
        pos_enc_dim: int,
        epsilon_init: float = 0.3,
    ):
        super().__init__()
        enc_s_dim = 3 + 2 * pos_enc_dim * 3

        self.attn_net = nn.Sequential(
            nn.Linear(enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, n_block),
        )
        self.phi_mlp = nn.Sequential(
            nn.LayerNorm(enc_dim + enc_s_dim),
            nn.Linear(enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, 1),
        )
        # ε parameter (annealed externally; log for positivity)
        self.log_epsilon = nn.Parameter(torch.tensor(math.log(epsilon_init)))

    def forward(
        self,
        V_all: torch.Tensor,   # [B, n_block, N, enc_dim]
        pos_enc: torch.Tensor, # [B, N, enc_s_dim]
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:         # [B, N, 1]
        w = torch.softmax(self.attn_net(pos_enc).clamp(-30, 30), dim=-1)
        V_perm = V_all.permute(0, 2, 1, 3)
        V_agg  = (w.unsqueeze(-1) * V_perm).sum(2)
        phi    = self.phi_mlp(torch.cat([V_agg, pos_enc], dim=-1))          # [B, N, 1]

        if epsilon is None:
            epsilon = torch.exp(self.log_epsilon).clamp(1e-3, 1.0)
        alpha = torch.sigmoid(-phi / (epsilon + 1e-6))
        return alpha.clamp(0.0, 1.0)


class GenericHead(nn.Module):
    """Standard delta-prediction head for unspecified field types."""

    def __init__(self, enc_dim: int, n_block: int, pos_enc_dim: int):
        super().__init__()
        enc_s_dim = 3 + 2 * pos_enc_dim * 3

        self.attn_net = nn.Sequential(
            nn.Linear(enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, n_block),
        )
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(enc_dim + enc_s_dim),
            nn.Linear(enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, 1),
        )

    def forward(self, V_all: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn_net(pos_enc).clamp(-30, 30), dim=-1)
        V_perm = V_all.permute(0, 2, 1, 3)
        V_agg  = (w.unsqueeze(-1) * V_perm).sum(2)
        return self.delta_mlp(torch.cat([V_agg, pos_enc], dim=-1))


class SlotIndexedDecoder(nn.Module):
    """
    Slot-indexed decoder (LPBF_NO_v3_design.md §4.7).
    Same-type slots share head parameters.
    """

    HEAD_TYPES = ("temperature", "interface", "velocity", "generic")

    def __init__(
        self,
        K: int,
        slot_types_str: List[str],
        enc_dim: int,
        n_block: int,
        pos_enc_dim: int,
        max_temp_delta: float = 500.0,
    ):
        super().__init__()
        self.K = K
        self.slot_types_str = slot_types_str

        self.heads = nn.ModuleDict({
            "temperature": TemperatureHead(enc_dim, n_block, pos_enc_dim, max_temp_delta),
            "interface":   InterfaceHead(enc_dim, n_block, pos_enc_dim),
            "velocity":    GenericHead(enc_dim, n_block, pos_enc_dim),
            "generic":     GenericHead(enc_dim, n_block, pos_enc_dim),
        })

    def forward(
        self,
        V_all_branches: Dict[int, torch.Tensor],  # {slot_idx: [B, n_block, N, enc_dim]}
        pos_enc: torch.Tensor,                    # [B, N, enc_s_dim]
        active_indices: List[int],
        epsilon: Optional[float] = None,
    ) -> Dict[int, torch.Tensor]:
        """Returns {slot_idx: [B, N, 1]} for each active slot."""
        outputs = {}
        for k in active_indices:
            head_type = self.slot_types_str[k]
            head      = self.heads[head_type]
            if head_type == "interface":
                outputs[k] = head(V_all_branches[k], pos_enc, epsilon)
            else:
                outputs[k] = head(V_all_branches[k], pos_enc)
        return outputs


# =============================================================================
# Full LPBF-NO v3 Stage 1 Model
# =============================================================================

class Model(nn.Module):
    """
    LPBF-NO v3 — Stage 1 Full Model.

    Interface matches physgto_attnres_max.Model (autoregressive / forward)
    with the addition of slot_values / presence_mask / active_indices.

    Args:
        K:               number of field slots (len(field_slots))
        slot_types_str:  list of K strings ('temperature'|'interface'|'velocity'|'generic')
        slot_defaults:   list of K default fill values (usually 0.0)
        cond_dim:        condition vector dimension (from dataset)
        enc_dim:         latent feature dimension
        N_block:         number of mixer blocks
        n_head:          attention heads
        n_token:         factorized attention token count
        n_cross_tokens:  cross-field attention token count
        pos_enc_dim:     Fourier frequencies
        x_boost:         x-axis frequency boost (scan direction)
        space_size:      spatial dimension (3 for LPBF)
        d_type:          field-type embedding dimension
        n_src:           KAN source term branches
        dt:              default timestep
        stepper_scheme:  'euler' or 'delta'
        layer_scale_init: GatedResidual init scale
        kappa_init:      Beer-Lambert absorption coefficient init
    """

    def __init__(
        self,
        K: int = 3,
        slot_types_str: Optional[List[str]] = None,
        slot_defaults: Optional[List[float]] = None,
        cond_dim: int = 32,
        enc_dim: int = 128,
        N_block: int = 4,
        n_head: int = 4,
        n_token: int = 64,
        n_cross_tokens: int = 32,
        pos_enc_dim: int = 6,
        x_boost: int = 2,
        space_size: int = 3,
        d_type: int = 16,
        n_src: int = 4,
        dt: float = 2e-5,
        stepper_scheme: str = "euler",
        layer_scale_init: float = 1e-2,
        kappa_init: float = 1e3,
        max_temp_delta: float = 500.0,
        # Legacy / compatibility aliases
        in_dim: int = None,  # ignored (use K)
        out_dim: int = None,  # ignored (use K)
        n_fields: int = None,  # ignored (use K)
        spatial_dim: int = 10,  # kept for build_model compat, not used
        **kwargs,
    ):
        super().__init__()

        self.K             = K
        self.enc_dim       = enc_dim
        self.N_block       = N_block
        self.dt            = dt
        self.stepper_scheme = stepper_scheme
        self.pos_enc_dim   = pos_enc_dim
        self.x_boost       = x_boost

        # Default slot types (all generic if not provided)
        if slot_types_str is None:
            slot_types_str = ["generic"] * K
        self.slot_types_str = slot_types_str

        # Integer type indices for embedding
        _type_map = {"temperature": 0, "interface": 1, "velocity": 2, "generic": 3}
        self.slot_type_indices = torch.tensor(
            [_type_map.get(t, 3) for t in slot_types_str], dtype=torch.long
        )

        # ---- Laser Driving Field module ----
        self.laser_field = LaserDrivingField(cond_dim, enc_dim, kappa_init)

        # ---- Slot Encoder ----
        self.encoder = SlotEncoder(
            K=K,
            enc_dim=enc_dim,
            space_size=space_size,
            pos_enc_dim=pos_enc_dim,
            x_boost=x_boost,
            d_type=d_type,
            n_type=4,
        )

        # Per-slot linear projections (only used when n_active > 1)
        # K projections pre-allocated; inactive ones are never called
        self.slot_projections = nn.ModuleList([
            nn.Linear(enc_dim, enc_dim) for _ in range(K)
        ])

        # ---- Mixer Blocks ----
        self.blocks = nn.ModuleList([
            LPBFMixerBlock(
                K=K,
                enc_dim=enc_dim,
                n_heads=n_head,
                n_tokens=n_token,
                n_cross_tokens=n_cross_tokens,
                n_src=n_src,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(N_block)
        ])

        # ---- Decoder ----
        self.decoder = SlotIndexedDecoder(
            K=K,
            slot_types_str=slot_types_str,
            enc_dim=enc_dim,
            n_block=N_block,
            pos_enc_dim=pos_enc_dim,
            max_temp_delta=max_temp_delta,
        )

        # epsilon schedule state (can be set externally by trainer)
        self.interface_epsilon: float = 0.3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_time(self, time_i: torch.Tensor, dt_tensor: torch.Tensor) -> torch.Tensor:
        """Mirrors physgto_attnres_max._encode_time."""
        time_info = torch.cat([time_i, dt_tensor], dim=-1)
        t_fourier = fourier_embedding(time_info, 0, self.pos_enc_dim)
        t_low     = fourier_embedding(time_i, 0, 2)
        return torch.cat([t_fourier, t_low], dim=-1)

    def _get_pos_enc(self, node_pos: torch.Tensor) -> torch.Tensor:
        return fourier_embedding_pos(node_pos, self.pos_enc_dim, self.x_boost)

    def _extract_grid_shape(self, spatial_inform: torch.Tensor) -> Tuple[int, int, int]:
        """Extract (Hx, Hy, Hz) from spatial_inform [B, 10]."""
        ds = spatial_inform[0, 6:9].long()  # [3] — same for all in batch
        return (int(ds[0]), int(ds[1]), int(ds[2]))

    # ------------------------------------------------------------------
    # Single-step forward
    # ------------------------------------------------------------------

    def forward(
        self,
        state_in: torch.Tensor,          # [B, N, K]
        node_pos: torch.Tensor,          # [B, N, 3] normalized
        time_i: torch.Tensor,            # [B, 1]
        conditions: torch.Tensor,        # [B, cond_dim]
        spatial_inform: torch.Tensor,    # [B, 10]
        presence_mask: torch.Tensor,     # [B, K]
        active_indices: List[int],
        node_pos_physical: Optional[torch.Tensor] = None,  # [B, N, 3]
        dt: Optional[float] = None,
        pos_enc: Optional[torch.Tensor] = None,
        # Legacy signature compatibility (ignored)
        edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns next state [B, N, K]."""
        B    = state_in.shape[0]
        device = state_in.device

        # ---- Time encoding ----
        if len(time_i.shape) == 1:
            time_i = time_i.view(-1, 1)
        _dt = dt if dt is not None else self.dt
        if isinstance(_dt, (float, int)):
            dt_t = torch.full((B, 1), float(_dt), dtype=time_i.dtype, device=device)
        else:
            dt_t = torch.as_tensor(_dt, dtype=time_i.dtype, device=device).view(B, 1)
        t_enc = self._encode_time(time_i, dt_t)  # [B, enc_t_dim]
        t_enc_n = t_enc.unsqueeze(1).expand(B, state_in.shape[1], -1)  # [B, N, enc_t_dim]

        # ---- Position encoding ----
        if pos_enc is None:
            pos_enc = self._get_pos_enc(node_pos)   # [B, N, enc_s_dim]

        # ---- Grid shape ----
        grid_shape = self._extract_grid_shape(spatial_inform)

        # ---- Laser driving field ----
        phys_pos = node_pos_physical if node_pos_physical is not None else node_pos
        # Find alpha.air slot for Beer-Lambert (slot type == interface, index 0 preferred)
        alpha_air = None
        for si in active_indices:
            if self.slot_types_str[si] == "interface":
                # Use the current (normalized) VoF value — VoF (0,1) so no denorm needed
                alpha_air = state_in[..., si:si+1]
                break

        L_field, laser_gamma, laser_beta, delta_x = self.laser_field(
            phys_pos, conditions, spatial_inform, alpha_air
        )

        # ---- Slot encoder → shared representation ----
        slot_type_idx = self.slot_type_indices.to(device)
        h_shared = self.encoder(
            node_pos      = node_pos,
            delta_x_laser = delta_x,
            L_field       = L_field,
            time_emb      = t_enc_n,
            slot_values   = state_in,
            presence_mask = presence_mask,
            slot_type_indices = slot_type_idx,
            laser_gamma   = laser_gamma,
            laser_beta    = laser_beta,
        )  # [B, N, enc_dim]

        # ---- Per-slot projections (skip if single active slot) ----
        if len(active_indices) == 1:
            h_branches: Dict[int, torch.Tensor] = {active_indices[0]: h_shared}
        else:
            h_branches = {k: self.slot_projections[k](h_shared) for k in active_indices}

        # ---- Mixer blocks ----
        blocks_history: Dict[int, List[torch.Tensor]] = {k: [h_branches[k]] for k in active_indices}
        V_all_branches: Dict[int, List[torch.Tensor]] = {k: [] for k in active_indices}

        for blk in self.blocks:
            h_branches, blocks_history = blk(
                h_branches    = h_branches,
                L_field       = L_field,
                grid_shape    = grid_shape,
                blocks_history = blocks_history,
                slot_type_strs = self.slot_types_str,
                active_indices = active_indices,
            )
            for k in active_indices:
                V_all_branches[k].append(h_branches[k])

        # Stack: {k: [B, N_block, N, enc_dim]}
        V_all_stacked = {k: torch.stack(V_all_branches[k], dim=1) for k in active_indices}

        # ---- Decoder ----
        deltas = self.decoder(V_all_stacked, pos_enc, active_indices, self.interface_epsilon)

        # ---- Assemble output ----
        state_out = state_in.clone()
        if self.stepper_scheme == "euler":
            for k in active_indices:
                field_type = self.slot_types_str[k]
                if field_type == "interface":
                    # InterfaceHead returns absolute α (not delta)
                    state_out[..., k:k+1] = deltas[k]
                else:
                    state_out[..., k:k+1] = state_in[..., k:k+1] + deltas[k] * _dt
        else:
            for k in active_indices:
                field_type = self.slot_types_str[k]
                if field_type == "interface":
                    state_out[..., k:k+1] = deltas[k]
                else:
                    state_out[..., k:k+1] = state_in[..., k:k+1] + deltas[k]

        return state_out  # [B, N, K]

    # ------------------------------------------------------------------
    # Autoregressive rollout (compatible with main_v2.py)
    # ------------------------------------------------------------------

    def autoregressive(
        self,
        state_in: torch.Tensor,           # [B, N, K]
        node_pos: torch.Tensor,           # [B, N, 3]
        edges: torch.Tensor,              # [B, ne, 2]  (kept for compat, unused)
        time_seq: torch.Tensor,           # [B, T, 1]
        spatial_inform: torch.Tensor,     # [B, 10]
        conditions: torch.Tensor,         # [B, cond_dim]
        dt=None,
        check_point=False,
        teacher_forcing: bool = False,
        gt_states: Optional[torch.Tensor] = None,
        # Slot-specific (required for LPBF-NO v3)
        presence_mask: Optional[torch.Tensor] = None,   # [B, K]
        active_indices: Optional[List[int]] = None,
        node_pos_physical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns predicted states [B, T, N, K].
        presence_mask and active_indices default to "all K slots active"
        when not provided (backward-compatible).
        """
        B, N, K = state_in.shape
        device   = state_in.device
        T        = time_seq.shape[1]

        # Default: treat all slots as active
        if presence_mask is None:
            presence_mask = torch.ones(B, K, device=device)
        if active_indices is None:
            active_indices = list(range(K))

        # Pre-compute position encoding (shared across timesteps)
        pos_enc = self._get_pos_enc(node_pos)

        state_t  = state_in
        outputs  = []

        for t in range(T):
            time_i = time_seq[:, t]  # [B, 1]

            def _forward(s, ti):
                return self.forward(
                    state_in          = s,
                    node_pos          = node_pos,
                    time_i            = ti,
                    conditions        = conditions,
                    spatial_inform    = spatial_inform,
                    presence_mask     = presence_mask,
                    active_indices    = active_indices,
                    node_pos_physical = node_pos_physical,
                    dt                = dt,
                    pos_enc           = pos_enc,
                )

            if check_point is True or (isinstance(check_point, int) and t >= check_point):
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t = state_t.detach().requires_grad_(True)
                state_pred = checkpoint(_forward, state_t, time_i, use_reentrant=False)
            else:
                state_pred = _forward(state_t, time_i)

            outputs.append(state_pred)

            if t < T - 1:
                if teacher_forcing and gt_states is not None:
                    state_t = gt_states[:, t]
                else:
                    state_t = state_pred

        return torch.stack(outputs, dim=1)  # [B, T, N, K]


# =============================================================================
# Quick validation
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 70)
    print("LPBF-NO v3 Stage 1 — Quick Shape Validation")
    print("=" * 70)

    B, N, K = 2, 8 * 6 * 4, 3
    # 3-slot config: T, alpha.air, gamma_liquid
    slot_types = ["temperature", "interface", "interface"]
    active_idx  = [0, 1]   # only T and alpha.air active

    model = Model(
        K=K,
        slot_types_str=slot_types,
        cond_dim=8,
        enc_dim=32,
        N_block=2,
        n_head=4,
        n_token=16,
        n_cross_tokens=8,
        pos_enc_dim=3,
        n_src=2,
        dt=2e-5,
    )

    state    = torch.randn(B, N, K)
    node_pos = torch.rand(B, N, 3)
    node_pos_phys = node_pos * 1e-3   # physical units ~mm
    time_i   = torch.full((B, 1), 1e-5)
    cond     = torch.randn(B, 8)

    # spatial_inform: 6 bounds + 3 ds_shape + 1 time_ref
    si       = torch.zeros(B, 10)
    si[:, :6] = torch.tensor([0, 1e-3, -5e-4, 0, 0, 5e-4])
    si[:, 6:9] = torch.tensor([8, 6, 4], dtype=torch.float32)  # Hx, Hy, Hz
    si[:, 9]   = 1.0

    pm = torch.zeros(B, K)
    pm[:, active_idx] = 1.0

    # Single step
    out1 = model(
        state_in=state, node_pos=node_pos, time_i=time_i,
        conditions=cond, spatial_inform=si,
        presence_mask=pm, active_indices=active_idx,
        node_pos_physical=node_pos_phys,
    )
    print(f"[single step] out: {out1.shape}  (expect {(B, N, K)})")
    assert out1.shape == (B, N, K)

    # Autoregressive
    T = 3
    time_seq = torch.linspace(0, 6e-5, T).view(1, T, 1).expand(B, -1, -1)
    edges    = torch.zeros(B, 1, 2, dtype=torch.long)  # dummy
    out_ar   = model.autoregressive(
        state_in=state, node_pos=node_pos, edges=edges,
        time_seq=time_seq, spatial_inform=si, conditions=cond,
        presence_mask=pm, active_indices=active_idx,
        node_pos_physical=node_pos_phys,
    )
    print(f"[autoregressive] out: {out_ar.shape}  (expect {(B, T, N, K)})")
    assert out_ar.shape == (B, T, N, K)

    # Backward
    model.train()
    loss = out_ar.sum()
    loss.backward()
    print("[backward] passed ✓")

    # Single-slot (n_active = 1)
    model1 = Model(K=1, slot_types_str=["temperature"], cond_dim=8, enc_dim=32,
                   N_block=2, n_head=4, n_token=16, n_cross_tokens=8, pos_enc_dim=3, n_src=2)
    st1 = torch.randn(B, N, 1)
    pm1 = torch.ones(B, 1)
    si1 = si.clone(); si1[:, 6:9] = torch.tensor([8, 6, 4], dtype=torch.float)
    out1f = model1(state_in=st1, node_pos=node_pos, time_i=time_i,
                   conditions=cond, spatial_inform=si1,
                   presence_mask=pm1, active_indices=[0],
                   node_pos_physical=node_pos_phys)
    print(f"[single-slot T] out: {out1f.shape}  (expect {(B, N, 1)})")
    assert out1f.shape == (B, N, 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters (K=3, enc=32): {n_params/1e3:.1f}K")
    print("=" * 70)
    print("✅  All shape checks passed!")
    print("=" * 70)
