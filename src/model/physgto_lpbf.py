"""
physgto_lpbf.py — PhysGTO-LPBF: Physics-Informed Neural Operator for LPBF

Architecture:
  1. LaserFieldModule     — Gaussian laser intensity per node (physics-based)
  2. ScaleAwareEncoder    — dual pos encoding (norm+abs) + triple time + spatially-varying FiLM
  3. MultiFieldMixer      — reused from v3 (Block AttnRes + Multi-Field Cross-Attention)
  4. SourceTermDecoder    — physics-structured branches with graceful degradation:
       A. DiffusionBranch           (heat diffusion via GNN neighbor differencing)
       B. LaserSourceBranch         (absorbed laser power, always positive)
       C. RadiationBranch           (surface radiation, T^4 structure)
       D. LatentHeatBranch          (phase-change energy)
       E. RecoilPressureBranch      (metal vapor recoil, exponential T dependence)
       F. EvaporationBranch         (surface evaporation heat loss)
       G. ResidualBranch            (convection + Marangoni + unmodelled physics)
       H. InterfaceEvolutionBranch  (VoF interface motion, only when alpha predicted)
       I. InterfaceSharpeningBranch (pushes alpha toward 0/1)
     + SpatialGating          (per-node softmax weights, not global scalars)
  5. InterfaceGNN (optional) — interface-proximity-weighted message passing

Graceful degradation:
  Each branch checks field availability at __init__ time (has_T, has_alpha, has_gamma)
  and selects the best available operating mode automatically.

Registered as model name "gto_lpbf".
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_softmax, scatter_add

from src.model.physgto_attnres_multi_v3 import (
    MLP,
    GNN,
    Atten,
    FieldCrossAttention,
    RMSNorm,
    block_attn_res,
    FourierEmbedding,
    FourierEmbedding_pos,
    get_edge_info,
    MultiFieldMixer,
    MultiFieldDecoder,
)


# =============================================================================
# Default physics constants (Ti-6Al-4V)
# =============================================================================

DEFAULT_PHYSICS_PARAMS = {
    "T_ref": 300.0,
    "T_solidus": 1877.0,
    "T_liquidus": 1923.0,
    "T_vaporization": 3133.0,
    "distribution_factor": 3.0,
    "emissivity": 0.4,
    "latent_heat_vap": 9.83e6,    # J/kg
    "molar_mass": 0.0479,          # kg/mol (Ti)
    "gas_constant": 8.314,         # J/(mol·K)
}


# =============================================================================
# Module 1: Laser Field Computer
# =============================================================================

class LaserFieldModule(nn.Module):
    """
    Computes per-node Gaussian laser intensity from raw physical parameters.

    Inputs (all in physical units, NOT normalised):
        node_pos_abs  [bs, N, 3]  — absolute node coords (m)
        laser_pos     [bs, 3]     — laser xyz at current timestep (m)
        laser_params  [bs, 4]     — [P_L (W), r (m), absorptivity, V_scan (m/s)]
        alpha_air     [bs, N, 1] or None  — VoF air field (for surface attenuation)
        spatial_inform[bs, 10]   — bounds + ds_shape + time_ref

    Outputs:
        laser_field   [bs, N, 1]  — normalised intensity in [0, 1]
        laser_feat    [bs, N, D]  — learnable enrichment features
    """

    def __init__(self, d_laser: int = 32, sigma: float = 0.05,
                 distribution_factor: float = 3.0, kappa_init: float = 5.0):
        super().__init__()
        self.d_laser = d_laser
        self.sigma = sigma
        self.distribution_factor = distribution_factor
        self.kappa = nn.Parameter(torch.tensor(kappa_init))

        self.enrich_mlp = MLP(input_size=5, output_size=d_laser,
                              hidden_size=d_laser, n_hidden=2,
                              act='SiLU', layer_norm=True)

    def forward(
        self,
        node_pos_abs: torch.Tensor,
        laser_pos: torch.Tensor,
        laser_params: torch.Tensor,
        spatial_inform: torch.Tensor,
        alpha_air: Optional[torch.Tensor] = None,
        dt_since: Optional[torch.Tensor] = None,
    ):
        bs, N, _ = node_pos_abs.shape
        P_L = laser_params[:, 0:1]
        r = laser_params[:, 1:2].clamp(min=1e-9)
        absorptivity = laser_params[:, 2:3]

        x_laser = laser_pos[:, 0:1]
        y_laser = laser_pos[:, 1:2]
        z_laser = laser_pos[:, 2:3]

        x = node_pos_abs[..., 0]
        y = node_pos_abs[..., 1]
        z = node_pos_abs[..., 2]

        dx = x - x_laser
        dz = z - z_laser
        d2 = dx ** 2 + dz ** 2

        r_sq = r ** 2
        I_raw = (P_L / (math.pi * r_sq + 1e-30)) * torch.exp(
            -self.distribution_factor * d2 / (r_sq + 1e-30)
        )

        I_max = I_raw.max(dim=-1, keepdim=True).values.clamp(min=1e-10)
        I_norm = I_raw / I_max

        # Surface attenuation: Beer-Lambert depth decay + interface peak
        # alpha_air ≈ 1 in pure air (transparent), ≈ 0 in pure metal (opaque)
        # Absorption peaks at the air-metal interface (alpha ≈ 0.5)
        if alpha_air is not None:
            alpha = alpha_air.squeeze(-1)
            # Quadratic peak at alpha=0.5: 4*α*(1-α) ∈ [0, 1], max at α=0.5
            surface_factor = 4.0 * alpha * (1.0 - alpha)
            # Beer-Lambert depth proxy: attenuate in metal bulk (low alpha)
            depth_atten = torch.exp(-self.kappa * (1.0 - alpha).clamp(min=0.0))
            I_att = I_norm * (surface_factor + depth_atten) * 0.5
        else:
            y_max = spatial_inform[:, 3:4]
            y_surface = y_max * 0.7
            I_att = I_norm * torch.sigmoid((y - y_surface) / self.sigma)

        # Apply absorptivity
        I_att = I_att * absorptivity

        # Learnable enrichment: 5 physics-informed scalar features per node
        d_norm = torch.sqrt(d2 + 1e-12) / (r + 1e-9)
        dy_norm = (y - y_laser) / (r + 1e-9)

        if dt_since is None:
            dt_feat = torch.zeros(bs, N, device=node_pos_abs.device)
        else:
            dt_feat = dt_since.view(bs, 1).expand(-1, N)

        feat_raw = torch.stack(
            [I_att, d_norm, dy_norm, dt_feat, absorptivity.expand(-1, N)], dim=-1
        )
        laser_feat = self.enrich_mlp(feat_raw)

        return I_att.unsqueeze(-1), laser_feat


# =============================================================================
# Module 2: Scale-Aware Encoder
# =============================================================================

class ScaleAwareEncoder(nn.Module):
    """
    Dual position encoding + triple time + spatially-varying FiLM from laser.
    """

    def __init__(
        self,
        space_size: int = 3,
        n_fields: int = 2,
        enc_dim: int = 128,
        enc_t_dim: int = 11,
        cond_dim: int = 32,
        spatial_dim: int = 10,
        pos_enc_dim: int = 5,
        pos_x_boost: int = 2,
        d_laser: int = 32,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.pos_enc_dim = pos_enc_dim
        self.pos_x_boost = pos_x_boost

        _single_pos_dim = space_size + 2 * pos_enc_dim * space_size
        enc_s_dim_double = _single_pos_dim * 2
        self.enc_s_dim = enc_s_dim_double

        _t_abs_dim = 1 + 2 * 3
        self.enc_t_total = enc_t_dim + _t_abs_dim

        self.fv_fields = nn.ModuleList([
            MLP(input_size=1 + space_size, output_size=enc_dim,
                act='SiLU', layer_norm=False)
            for _ in range(n_fields)
        ])

        self.fv_time = MLP(input_size=self.enc_t_total, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_cond = MLP(input_size=cond_dim, output_size=enc_dim, act='SiLU', layer_norm=False)
        self.fv_spatial = MLP(input_size=spatial_dim, output_size=enc_dim, act='SiLU', layer_norm=False)

        self.fuse_para = MLP(input_size=enc_dim * 3, output_size=enc_dim * 2,
                             act='SiLU', layer_norm=False)

        self.laser_film_net = nn.Linear(d_laser, enc_dim * 2, bias=True)
        nn.init.zeros_(self.laser_film_net.weight)
        nn.init.zeros_(self.laser_film_net.bias)

        self.fe = MLP(input_size=2 * space_size + 1, output_size=enc_dim,
                      n_hidden=1, act='SiLU', layer_norm=False)

    def forward(
        self,
        node_pos_norm: torch.Tensor,
        node_pos_abs: torch.Tensor,
        state_in: torch.Tensor,
        time_enc: torch.Tensor,
        conditions: torch.Tensor,
        edges: torch.Tensor,
        spatial_inform: torch.Tensor,
        laser_feat: torch.Tensor,
    ):
        pos_enc_norm = FourierEmbedding_pos(node_pos_norm, self.pos_enc_dim, self.pos_x_boost)

        ref_x = (spatial_inform[:, 1] - spatial_inform[:, 0]).clamp(min=1e-6)
        ref_y = (spatial_inform[:, 3] - spatial_inform[:, 2]).clamp(min=1e-6)
        ref_z = (spatial_inform[:, 5] - spatial_inform[:, 4]).clamp(min=1e-6)
        ref_scale = torch.stack([ref_x, ref_y, ref_z], dim=-1).unsqueeze(1)
        pos_abs_scaled = node_pos_abs / (ref_scale + 1e-9)
        pos_enc_abs = FourierEmbedding_pos(pos_abs_scaled, self.pos_enc_dim, self.pos_x_boost)

        pos_enc = torch.cat([pos_enc_norm, pos_enc_abs], dim=-1)

        time_enc_v = self.fv_time(time_enc)
        cond_enc = self.fv_cond(conditions)
        spatial_enc = self.fv_spatial(spatial_inform)
        h = torch.cat([cond_enc, time_enc_v, spatial_enc], dim=-1)
        para = self.fuse_para(h)
        gamma_g, beta_g = para.chunk(2, dim=-1)

        laser_para = self.laser_film_net(laser_feat)
        gamma_l, beta_l = laser_para.chunk(2, dim=-1)

        V_list = []
        for i in range(self.n_fields):
            field_i = state_in[..., i:i+1]
            inp = torch.cat([node_pos_norm, field_i], dim=-1)
            V_i = self.fv_fields[i](inp)
            V_i = V_i * gamma_g.unsqueeze(1) + beta_g.unsqueeze(1)
            V_i = V_i * (1.0 + gamma_l) + beta_l
            V_list.append(V_i)

        E = self.fe(get_edge_info(edges, node_pos_norm))
        return V_list, E, pos_enc


# =============================================================================
# Module 3: Source-Term Branches
# =============================================================================

class DiffusionBranch(nn.Module):
    """
    Branch A: Heat diffusion via GNN-style neighbor differencing.
    Includes phase-dependent effective conductivity when gamma is available.
    """

    def __init__(self, enc_dim: int, edge_raw_dim: int, has_T: bool, has_gamma: bool):
        super().__init__()
        self.has_T = has_T
        self.has_gamma = has_gamma
        in_msg = 1 + edge_raw_dim + (1 if has_gamma else 0)
        self.msg_mlp = MLP(input_size=in_msg, output_size=enc_dim,
                           hidden_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=True)
        self.node_mlp = MLP(input_size=enc_dim + enc_dim, output_size=enc_dim,
                            hidden_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

    def forward(self, V_field, edges, edge_raw, T_field=None, gamma_field=None):
        bs, N, _ = V_field.shape
        s_idx = edges[..., 0]
        r_idx = edges[..., 1]

        if self.has_T and T_field is not None:
            src = T_field.squeeze(-1)
        else:
            src = V_field.mean(dim=-1)

        s_vals = torch.gather(src, 1, s_idx)
        r_vals = torch.gather(src, 1, r_idx)
        diff = (r_vals - s_vals).unsqueeze(-1)

        msg_parts = [diff, edge_raw]

        # Phase-dependent conductivity modulation
        if self.has_gamma and gamma_field is not None:
            gamma_s = torch.gather(gamma_field.squeeze(-1), 1, s_idx).unsqueeze(-1)
            gamma_r = torch.gather(gamma_field.squeeze(-1), 1, r_idx).unsqueeze(-1)
            # Effective conductivity ratio: liquid has higher k → larger messages
            k_eff = 0.5 * (gamma_s + gamma_r)
            msg_parts.append(k_eff)

        msg_in = torch.cat(msg_parts, dim=-1)
        msg = self.msg_mlp(msg_in)

        col = r_idx.unsqueeze(-1).expand_as(msg)
        agg = scatter_add(msg, col, dim=1, dim_size=N)

        h = self.node_mlp(torch.cat([V_field, agg], dim=-1))
        return self.out_proj(h)


class LaserSourceBranch(nn.Module):
    """Branch B: Absorbed laser power. Output ≥ 0 (softplus)."""

    def __init__(self, enc_dim: int, d_laser: int, has_T: bool):
        super().__init__()
        self.has_T = has_T
        cond_dim = enc_dim
        self.mlp = MLP(input_size=d_laser + cond_dim, output_size=enc_dim,
                       hidden_size=enc_dim, n_hidden=2, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

    def forward(self, laser_feat, V_field, T_field=None):
        if self.has_T and T_field is not None:
            cond = T_field.expand(-1, -1, V_field.shape[-1])
        else:
            cond = V_field
        h = self.mlp(torch.cat([laser_feat, cond], dim=-1))
        return F.softplus(self.out_proj(h))


class RadiationBranch(nn.Module):
    """
    Branch C: Surface radiation Q_rad ∝ -(T^4 - T_ref^4) * |∇α|
    Configurable T_ref.
    """

    def __init__(self, enc_dim: int, edge_raw_dim: int, has_T: bool, has_alpha: bool,
                 space_size: int = 3, T_ref: float = 300.0):
        super().__init__()
        self.has_T = has_T
        self.has_alpha = has_alpha
        self.T_ref = T_ref

        in_dim = enc_dim
        if has_T:
            in_dim += 1
        if has_alpha:
            in_dim += 1

        self.mlp = MLP(input_size=in_dim, output_size=enc_dim,
                       hidden_size=enc_dim, n_hidden=2, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

        if has_alpha:
            self.grad_mlp = MLP(input_size=1 + edge_raw_dim, output_size=1,
                                hidden_size=enc_dim // 2, n_hidden=1,
                                act='SiLU', layer_norm=False)

    def _compute_grad_alpha(self, alpha, edges, edge_raw):
        bs, N, _ = alpha.shape
        s_idx = edges[..., 0]
        r_idx = edges[..., 1]
        a_s = torch.gather(alpha.squeeze(-1), 1, s_idx)
        a_r = torch.gather(alpha.squeeze(-1), 1, r_idx)
        da = (a_r - a_s).abs().unsqueeze(-1)
        msg = self.grad_mlp(torch.cat([da, edge_raw], dim=-1))
        col = r_idx.unsqueeze(-1).expand_as(msg)
        grad = scatter_add(msg, col, dim=1, dim_size=N)
        return grad

    def forward(self, V_field, edges, edge_raw,
                T_field=None, alpha_field=None, spatial_inform=None):
        parts = [V_field]

        if self.has_T and T_field is not None:
            T_norm = T_field / 1000.0
            T4_term = T_norm ** 4 - (self.T_ref / 1000.0) ** 4
            parts.append(T4_term)

        if self.has_alpha and alpha_field is not None:
            grad_a = self._compute_grad_alpha(alpha_field, edges, edge_raw)
            parts.append(grad_a)

        h_in = torch.cat(parts, dim=-1)
        h = self.mlp(h_in)
        return -F.softplus(self.out_proj(h))


class LatentHeatBranch(nn.Module):
    """Branch D: Phase-change latent heat. Configurable solidus/liquidus."""

    def __init__(self, enc_dim: int, has_T: bool, has_gamma: bool,
                 T_solidus: float = 1877.0, T_liquidus: float = 1923.0):
        super().__init__()
        self.has_T = has_T
        self.has_gamma = has_gamma
        self.T_solidus = T_solidus
        self.T_liquidus = T_liquidus

        in_dim = enc_dim
        if has_gamma:
            in_dim += 1
        elif has_T:
            in_dim += 1

        self.mlp = MLP(input_size=in_dim, output_size=enc_dim,
                       hidden_size=enc_dim, n_hidden=2, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

    def forward(self, V_field, T_field=None, gamma_field=None):
        parts = [V_field]

        if self.has_gamma and gamma_field is not None:
            parts.append(gamma_field)
        elif self.has_T and T_field is not None:
            eps = self.T_liquidus - self.T_solidus + 1.0
            phase_proxy = torch.sigmoid((T_field - self.T_solidus) / eps)
            parts.append(phase_proxy)

        h = self.mlp(torch.cat(parts, dim=-1))
        return self.out_proj(h)


class RecoilPressureBranch(nn.Module):
    """
    Branch E: Recoil pressure from metal vaporization.
    Physics: P_recoil = 0.54·P₀·exp[LvM/R·(1/Tv - 1/T)]
    Only significant when T > T_vaporization.
    Structural bias: exponential activation on temperature ratio.
    """

    def __init__(self, enc_dim: int, has_T: bool, has_alpha: bool,
                 T_vaporization: float = 3133.0):
        super().__init__()
        self.has_T = has_T
        self.has_alpha = has_alpha
        self.T_vap = T_vaporization

        in_dim = enc_dim + (1 if has_T else 0) + (1 if has_alpha else 0)
        self.mlp = MLP(input_size=in_dim, output_size=enc_dim,
                       hidden_size=enc_dim, n_hidden=2, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

    def forward(self, V_field, T_field=None, alpha_field=None):
        parts = [V_field]

        if self.has_T and T_field is not None:
            # Exponential temperature structure: significant only near/above T_vap
            T_ratio = torch.clamp(T_field / self.T_vap, max=2.0)
            # exp((1 - 1/T_ratio)) peaks when T >> T_vap
            recoil_proxy = torch.exp(torch.clamp(1.0 - 1.0 / (T_ratio + 1e-6), max=5.0)) - 1.0
            parts.append(recoil_proxy)

        if self.has_alpha and alpha_field is not None:
            parts.append(alpha_field)

        h = self.mlp(torch.cat(parts, dim=-1))
        return self.out_proj(h)


class EvaporationBranch(nn.Module):
    """
    Branch F: Surface evaporation heat loss.
    Physics: Q_evap ∝ exp[LvM(T-Tv)/(RTTv)] · |∇α|
    Output sign: negative (heat loss from evaporation).
    """

    def __init__(self, enc_dim: int, edge_raw_dim: int, has_T: bool, has_alpha: bool,
                 T_vaporization: float = 3133.0):
        super().__init__()
        self.has_T = has_T
        self.has_alpha = has_alpha
        self.T_vap = T_vaporization

        in_dim = enc_dim + (1 if has_T else 0) + (1 if has_alpha else 0)
        self.mlp = MLP(input_size=in_dim, output_size=enc_dim,
                       hidden_size=enc_dim, n_hidden=2, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

        if has_alpha:
            self.grad_mlp = MLP(input_size=1 + edge_raw_dim, output_size=1,
                                hidden_size=enc_dim // 2, n_hidden=1,
                                act='SiLU', layer_norm=False)

    def _compute_grad_alpha(self, alpha, edges, edge_raw):
        bs, N, _ = alpha.shape
        s_idx = edges[..., 0]
        r_idx = edges[..., 1]
        a_s = torch.gather(alpha.squeeze(-1), 1, s_idx)
        a_r = torch.gather(alpha.squeeze(-1), 1, r_idx)
        da = (a_r - a_s).abs().unsqueeze(-1)
        msg = self.grad_mlp(torch.cat([da, edge_raw], dim=-1))
        col = r_idx.unsqueeze(-1).expand_as(msg)
        return scatter_add(msg, col, dim=1, dim_size=N)

    def forward(self, V_field, edges, edge_raw, T_field=None, alpha_field=None):
        parts = [V_field]

        if self.has_T and T_field is not None:
            # Evaporation is exponentially activated near T_vap
            T_ratio = (T_field - self.T_vap) / self.T_vap
            evap_proxy = torch.sigmoid(T_ratio * 10.0)  # smooth step near T_vap
            parts.append(evap_proxy)

        if self.has_alpha and alpha_field is not None:
            grad_a = self._compute_grad_alpha(alpha_field, edges, edge_raw)
            parts.append(grad_a)

        h = self.mlp(torch.cat(parts, dim=-1))
        return -F.softplus(self.out_proj(h))


class ResidualBranch(nn.Module):
    """Branch G: Flexible residual (convection, Marangoni, unmodelled physics)."""

    def __init__(self, enc_dim: int):
        super().__init__()
        self.mlp = MLP(input_size=enc_dim, output_size=enc_dim,
                       hidden_size=enc_dim, n_hidden=2, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

    def forward(self, V_field):
        return self.out_proj(self.mlp(V_field))


class InterfaceEvolutionBranch(nn.Module):
    """Branch H: VoF interface motion (only when alpha predicted)."""

    def __init__(self, enc_dim: int, edge_raw_dim: int, has_T: bool):
        super().__init__()
        self.has_T = has_T
        in_dim = 1 + edge_raw_dim + enc_dim
        self.msg_mlp = MLP(input_size=in_dim, output_size=enc_dim,
                           hidden_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=True)
        self.node_mlp = MLP(input_size=enc_dim + enc_dim, output_size=enc_dim,
                            hidden_size=enc_dim, n_hidden=1, act='SiLU', layer_norm=True)
        self.out_proj = nn.Linear(enc_dim, 1)

    def forward(self, V_field, alpha_field, edges, edge_raw, T_field=None):
        bs, N, _ = V_field.shape
        s_idx = edges[..., 0]
        r_idx = edges[..., 1]

        a_s = torch.gather(alpha_field.squeeze(-1), 1, s_idx).unsqueeze(-1)
        a_r = torch.gather(alpha_field.squeeze(-1), 1, r_idx).unsqueeze(-1)
        da = a_r - a_s

        cond_s = torch.gather(V_field, 1,
                              s_idx.unsqueeze(-1).expand(-1, -1, V_field.shape[-1]))

        msg = self.msg_mlp(torch.cat([da, edge_raw, cond_s], dim=-1))
        col = r_idx.unsqueeze(-1).expand_as(msg)
        agg = scatter_add(msg, col, dim=1, dim_size=N)
        h = self.node_mlp(torch.cat([V_field, agg], dim=-1))
        return self.out_proj(h)


class InterfaceSharpeningBranch(nn.Module):
    """
    Branch I: Pushes alpha toward 0 or 1 away from interface.
    Includes V_field dependence (flow affects interface sharpness).
    """

    def __init__(self, enc_dim: int):
        super().__init__()
        self.k_net = nn.Sequential(
            nn.Linear(enc_dim + 1, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, 1),
            nn.Softplus()
        )

    def forward(self, V_field, alpha_field):
        h = torch.cat([V_field, alpha_field], dim=-1)
        k = self.k_net(h) + 1.0
        correction = torch.sigmoid(k * (alpha_field - 0.5)) * 2.0 - 1.0
        return correction


# =============================================================================
# Spatial Gating
# =============================================================================

class SpatialGating(nn.Module):
    """Per-node gate weights. Initialised near uniform."""

    def __init__(self, n_branches: int, enc_dim: int, d_laser: int, enc_s_dim: int):
        super().__init__()
        gate_in = enc_dim + d_laser + enc_s_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, n_branches),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, V_field, laser_feat, pos_enc):
        gate_in = torch.cat([V_field, laser_feat, pos_enc], dim=-1)
        logits = self.gate_net(gate_in)
        return torch.softmax(logits, dim=-1)


# =============================================================================
# SourceTermDecoder Assembly
# =============================================================================

class SourceTermDecoder(nn.Module):
    """
    Physics-structured decoder combining all source-term branches.
    Aux losses: ortho_loss, balance_loss (stored after forward).
    """

    def __init__(
        self,
        fields: List[str],
        enc_dim: int,
        d_laser: int,
        enc_s_dim: int,
        space_size: int = 3,
        physics_params: Optional[Dict] = None,
    ):
        super().__init__()
        pp = {**DEFAULT_PHYSICS_PARAMS, **(physics_params or {})}

        self.has_T = "T" in fields
        self.has_alpha = "alpha.air" in fields
        self.has_gamma = "gamma_liquid" in fields
        self.T_idx = fields.index("T") if self.has_T else None
        self.alpha_idx = fields.index("alpha.air") if self.has_alpha else None
        self.gamma_idx = fields.index("gamma_liquid") if self.has_gamma else None

        edge_raw_dim = 2 * space_size + 1

        # Core branches (always created)
        self.diffusion = DiffusionBranch(enc_dim, edge_raw_dim, self.has_T, self.has_gamma)
        self.laser_src = LaserSourceBranch(enc_dim, d_laser, self.has_T)
        self.radiation = RadiationBranch(enc_dim, edge_raw_dim, self.has_T, self.has_alpha,
                                         space_size, T_ref=pp["T_ref"])
        self.latent = LatentHeatBranch(enc_dim, self.has_T, self.has_gamma,
                                        T_solidus=pp["T_solidus"], T_liquidus=pp["T_liquidus"])
        self.recoil = RecoilPressureBranch(enc_dim, self.has_T, self.has_alpha,
                                            T_vaporization=pp["T_vaporization"])
        self.evaporation = EvaporationBranch(enc_dim, edge_raw_dim, self.has_T, self.has_alpha,
                                              T_vaporization=pp["T_vaporization"])
        self.residual = ResidualBranch(enc_dim)

        self.n_branches = 7  # A-G

        # VoF branches (only when alpha predicted)
        self.interface_evo = None
        self.interface_sharp = None
        if self.has_alpha:
            self.interface_evo = InterfaceEvolutionBranch(enc_dim, edge_raw_dim, self.has_T)
            self.interface_sharp = InterfaceSharpeningBranch(enc_dim)
            self.n_branches += 2

        self.gating = SpatialGating(self.n_branches, enc_dim, d_laser, enc_s_dim)

        self.ortho_loss = torch.tensor(0.0)
        self.balance_loss = torch.tensor(0.0)

    def forward(
        self,
        V_field: torch.Tensor,
        laser_feat: torch.Tensor,
        pos_enc: torch.Tensor,
        edges: torch.Tensor,
        node_pos: torch.Tensor,
        state_in: torch.Tensor,
        spatial_inform: torch.Tensor,
    ):
        bs, N, _ = V_field.shape

        T_field = state_in[..., self.T_idx:self.T_idx+1].detach() if self.has_T else None
        alpha_field = state_in[..., self.alpha_idx:self.alpha_idx+1].detach() if self.has_alpha else None
        gamma_field = state_in[..., self.gamma_idx:self.gamma_idx+1].detach() if self.has_gamma else None

        edge_raw = get_edge_info(edges, node_pos)

        # Evaluate all branches
        out_diff = self.diffusion(V_field, edges, edge_raw, T_field, gamma_field)
        out_laser = self.laser_src(laser_feat, V_field, T_field)
        out_rad = self.radiation(V_field, edges, edge_raw, T_field, alpha_field, spatial_inform)
        out_latent = self.latent(V_field, T_field, gamma_field)
        out_recoil = self.recoil(V_field, T_field, alpha_field)
        out_evap = self.evaporation(V_field, edges, edge_raw, T_field, alpha_field)
        out_resid = self.residual(V_field)

        branches = [out_diff, out_laser, out_rad, out_latent, out_recoil, out_evap, out_resid]

        if self.has_alpha and self.interface_evo is not None:
            out_evo = self.interface_evo(V_field, alpha_field, edges, edge_raw, T_field)
            out_sharp = self.interface_sharp(V_field, alpha_field)
            branches += [out_evo, out_sharp]

        # Spatial gating
        weights = self.gating(V_field, laser_feat, pos_enc)
        branch_stack = torch.cat(branches, dim=-1)
        delta = (branch_stack * weights).sum(dim=-1, keepdim=True)

        self._compute_aux_losses(branches, weights)

        return delta

    def _compute_aux_losses(self, branches, weights):
        bs, N, _ = branches[0].shape
        bstack = torch.cat(branches, dim=-1)
        eps = 1e-8
        # Normalize per-batch (dim=1 is node dimension)
        normed = bstack / (bstack.norm(dim=1, keepdim=True) + eps)
        gram = torch.einsum('bni,bnj->bij', normed, normed) / N
        eye = torch.eye(self.n_branches, device=gram.device).unsqueeze(0)
        self.ortho_loss = ((gram - eye) ** 2).mean()

        entropy = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
        self.balance_loss = -entropy


# =============================================================================
# Time encoding helper
# =============================================================================

def _encode_time_lpbf(time_i, dt_tensor, abs_time, pos_enc_dim):
    time_info = torch.cat([time_i, dt_tensor], dim=-1)
    t_fourier = FourierEmbedding(time_info, 0, pos_enc_dim)
    t_low = FourierEmbedding(time_i, 0, 2)
    t_abs_enc = FourierEmbedding(abs_time.unsqueeze(-1) if abs_time.dim() == 1 else abs_time,
                                  0, 3)
    return torch.cat([t_fourier, t_low, t_abs_enc], dim=-1)


# =============================================================================
# Full Model
# =============================================================================

class Model(nn.Module):
    """
    PhysGTO-LPBF  (model name: "gto_lpbf")

    Autoregressive neural operator with physics-informed source-term branches.
    """

    def __init__(
        self,
        fields: List[str],
        space_size: int = 3,
        pos_enc_dim: int = 5,
        cond_dim: int = 32,
        spatial_dim: int = 10,
        N_block: int = 4,
        in_dim: int = 2,
        out_dim: int = 2,
        enc_dim: int = 128,
        n_head: int = 4,
        n_token: int = 128,
        n_latent: int = 4,
        dt: float = 2e-5,
        stepper_scheme: str = "euler",
        pos_x_boost: int = 2,
        n_fields: Optional[int] = None,
        cross_attn_heads: int = 4,
        d_laser: int = 32,
        ortho_weight: float = 0.01,
        balance_weight: float = 0.05,
        sharp_weight: float = 0.01,
        physics_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.dt = dt
        self.stepper_scheme = stepper_scheme
        self.pos_enc_dim = pos_enc_dim
        self.pos_x_boost = pos_x_boost
        self.fields = fields
        self.n_fields = n_fields if n_fields is not None else in_dim
        self.d_laser = d_laser
        self.ortho_weight = ortho_weight
        self.balance_weight = balance_weight
        self.sharp_weight = sharp_weight
        self.has_alpha = "alpha.air" in fields
        self.physics_params = {**DEFAULT_PHYSICS_PARAMS, **(physics_params or {})}

        # === Laser Field Module ===
        self.laser_module = LaserFieldModule(
            d_laser=d_laser,
            distribution_factor=self.physics_params["distribution_factor"],
        )

        # === Scale-Aware Encoder ===
        enc_t_dim_v3 = (1 + 2 * pos_enc_dim) * 2 + (1 + 2 * 2)
        self.encoder = ScaleAwareEncoder(
            space_size=space_size,
            n_fields=self.n_fields,
            enc_dim=enc_dim,
            enc_t_dim=enc_t_dim_v3,
            cond_dim=cond_dim,
            spatial_dim=spatial_dim,
            pos_enc_dim=pos_enc_dim,
            pos_x_boost=pos_x_boost,
            d_laser=d_laser,
        )
        enc_s_dim = self.encoder.enc_s_dim

        # === Multi-Field Mixer (reused from v3) ===
        self.mixer = MultiFieldMixer(
            N_block=N_block,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
            cross_attn_heads=cross_attn_heads,
            n_latent=n_latent,
        )

        # === Multi-Field Decoder (base prediction) ===
        self.decoder = MultiFieldDecoder(
            N_block=N_block,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            n_fields=self.n_fields,
        )

        # === Source-Term Decoder (physics-informed correction) ===
        self.source_decoder = SourceTermDecoder(
            fields=fields,
            enc_dim=enc_dim,
            d_laser=d_laser,
            enc_s_dim=enc_s_dim,
            space_size=space_size,
            physics_params=self.physics_params,
        )

        # Source correction → all fields
        self.src_proj = nn.Linear(1, self.n_fields, bias=False)
        nn.init.zeros_(self.src_proj.weight)

        self.aux_losses = {}

        # Bounded output controls
        self.T_idx = fields.index("T") if "T" in fields else None
        self.alpha_idx = fields.index("alpha.air") if "alpha.air" in fields else None
        self.gamma_idx = fields.index("gamma_liquid") if "gamma_liquid" in fields else None
        self.max_delta_T = 500.0

    def _encode_time(self, time_i, dt_tensor, abs_time):
        return _encode_time_lpbf(time_i, dt_tensor, abs_time, self.pos_enc_dim)

    def forward(
        self,
        state_in: torch.Tensor,
        node_pos: torch.Tensor,
        edges: torch.Tensor,
        time_i: torch.Tensor,
        conditions: torch.Tensor,
        spatial_inform: torch.Tensor,
        pos_enc=None,
        dt=None,
        node_pos_abs: Optional[torch.Tensor] = None,
        laser_pos: Optional[torch.Tensor] = None,
        laser_params: Optional[torch.Tensor] = None,
        abs_time: Optional[torch.Tensor] = None,
        alpha_air_in: Optional[torch.Tensor] = None,
    ):
        if len(time_i.shape) == 1:
            time_i = time_i.view(-1, 1)
        bs = time_i.shape[0]
        device = time_i.device

        if dt is None:
            dt_tensor = torch.full((bs, 1), self.dt, dtype=time_i.dtype, device=device)
        elif isinstance(dt, (float, int)):
            dt_tensor = torch.full((bs, 1), float(dt), dtype=time_i.dtype, device=device)
        elif isinstance(dt, (np.floating, np.integer)):
            dt_tensor = torch.tensor([dt], dtype=time_i.dtype, device=device).view(bs, 1)
        else:
            dt_tensor = dt.view(bs, 1).to(dtype=time_i.dtype, device=device)

        if abs_time is None:
            abs_time = time_i.squeeze(-1)

        # Laser Field
        N = state_in.shape[1]
        if (node_pos_abs is not None and laser_pos is not None
                and laser_params is not None):
            alpha_for_laser = alpha_air_in
            laser_field, laser_feat = self.laser_module(
                node_pos_abs, laser_pos, laser_params, spatial_inform, alpha_for_laser
            )
        else:
            laser_field = torch.zeros(bs, N, 1, device=device)
            laser_feat = torch.zeros(bs, N, self.d_laser, device=device)

        # Time Encoding
        t_enc = self._encode_time(time_i, dt_tensor, abs_time)

        edges_long = edges.long() if edges.dtype != torch.long else edges

        # Scale-Aware Encoder
        node_pos_abs_enc = node_pos_abs if node_pos_abs is not None else node_pos
        V_list, E, pos_enc_out = self.encoder(
            node_pos, node_pos_abs_enc, state_in, t_enc,
            conditions, edges_long, spatial_inform, laser_feat
        )

        # Mixer
        V_all_list = self.mixer(V_list, E, edges_long, pos_enc_out)

        # Base Decoder
        v_pred = self.decoder(V_all_list, pos_enc_out)

        # Source-Term Correction
        # Aggregate last block output from all fields
        V_last = sum(V_all_list[f][:, -1] for f in range(len(V_all_list))) / len(V_all_list)
        src_delta_1 = self.source_decoder(
            V_last, laser_feat, pos_enc_out, edges_long,
            node_pos, state_in, spatial_inform
        )
        src_delta = self.src_proj(src_delta_1)

        combined = v_pred + src_delta

        # Auxiliary losses
        self.aux_losses = {
            "ortho": (self.source_decoder.ortho_loss, self.ortho_weight),
            "balance": (self.source_decoder.balance_loss, self.balance_weight),
        }

        if self.stepper_scheme == "euler":
            with autocast(device_type="cuda", enabled=False):
                delta = combined.float() * dt_tensor.unsqueeze(-1).float()
                # Bound temperature delta to prevent catastrophic spikes
                if self.T_idx is not None:
                    delta[..., self.T_idx] = torch.tanh(
                        delta[..., self.T_idx] / self.max_delta_T
                    ) * self.max_delta_T
                state_pred = state_in.float() + delta
        else:
            delta = combined
            if self.T_idx is not None:
                delta = delta.clone()
                delta[..., self.T_idx] = torch.tanh(
                    delta[..., self.T_idx] / self.max_delta_T
                ) * self.max_delta_T
            state_pred = state_in + delta

        # Clamp VoF fields to [0, 1]
        if self.alpha_idx is not None:
            state_pred[..., self.alpha_idx] = state_pred[..., self.alpha_idx].clamp(0.0, 1.0)
        if self.gamma_idx is not None:
            state_pred[..., self.gamma_idx] = state_pred[..., self.gamma_idx].clamp(0.0, 1.0)

        return state_pred

    def autoregressive(
        self,
        state_in,
        node_pos,
        edges,
        time_seq,
        spatial_inform,
        conditions,
        dt=None,
        check_point=False,
        teacher_forcing=False,
        gt_states=None,
        node_pos_abs=None,
        laser_params=None,
        laser_traj=None,
        abs_time_seq=None,
    ):
        state_t = state_in
        outputs = []
        T = time_seq.shape[1]

        for t in range(T):
            time_i = time_seq[:, t]

            laser_pos_t = laser_traj[:, t+1] if laser_traj is not None else None
            abs_time_t = abs_time_seq[:, t+1] if abs_time_seq is not None else None
            alpha_idx = self.fields.index("alpha.air") if self.has_alpha else None
            alpha_in_t = state_t[..., alpha_idx:alpha_idx+1].clone() if alpha_idx is not None else None

            def _fwd(s_t, t_i,
                     _lp=laser_pos_t, _at=abs_time_t, _ai=alpha_in_t):
                return self.forward(
                    s_t, node_pos, edges, t_i, conditions, spatial_inform, dt=dt,
                    node_pos_abs=node_pos_abs, laser_pos=_lp,
                    laser_params=laser_params, abs_time=_at,
                    alpha_air_in=_ai,
                )

            if check_point is True or (isinstance(check_point, int) and t >= check_point):
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t = state_t.detach().requires_grad_(True)
                state_pred = checkpoint(_fwd, state_t, time_i, use_reentrant=False)
            else:
                state_pred = _fwd(state_t, time_i)

            outputs.append(state_pred)

            if t < T - 1:
                state_t = gt_states[:, t] if (teacher_forcing and gt_states is not None) else state_pred

        return torch.stack(outputs, dim=1)


# =============================================================================
# Quick smoke test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print("PhysGTO-LPBF smoke test")
    print("=" * 60)

    bs, N, ne, T = 2, 64, 128, 4
    space_dim = 3

    for test_fields in [["T", "alpha.air"], ["T"], ["alpha.air"]]:
        n_f = len(test_fields)
        model = Model(
            fields=test_fields,
            space_size=space_dim,
            pos_enc_dim=3,
            cond_dim=8,
            spatial_dim=10,
            N_block=2,
            in_dim=n_f,
            out_dim=n_f,
            enc_dim=64,
            n_head=4,
            n_token=32,
            n_latent=2,
            n_fields=n_f,
            d_laser=16,
            dt=2e-5,
        )
        state_in = torch.randn(bs, N, n_f)
        node_pos = torch.rand(bs, N, space_dim)
        node_pos_abs = node_pos * 1e-3
        edges = torch.randint(0, N, (bs, ne, 2))
        time_seq = torch.linspace(0, 4e-4, T).unsqueeze(0).expand(bs, -1)
        conditions = torch.randn(bs, 8)
        spatial_inform = torch.cat([
            torch.tensor([0., 1e-3, 0., 1e-3, 0., 1e-3]).unsqueeze(0).expand(bs, -1),
            torch.ones(bs, 3) * 8,
            torch.ones(bs, 1) * 2e-5
        ], dim=-1)
        laser_params = torch.tensor([[200., 40e-6, 0.4, 0.3]]).expand(bs, -1)
        laser_traj = torch.zeros(bs, T+1, 3)
        abs_time_seq = torch.linspace(0, 4e-4, T+1).unsqueeze(0).expand(bs, -1)

        out = model.autoregressive(
            state_in, node_pos, edges, time_seq,
            spatial_inform, conditions, dt=2e-5,
            node_pos_abs=node_pos_abs, laser_params=laser_params,
            laser_traj=laser_traj, abs_time_seq=abs_time_seq,
        )
        assert out.shape == (bs, T, N, n_f), f"Shape mismatch: {out.shape}"
        print(f"  fields={test_fields}  out={out.shape}  aux_losses={list(model.aux_losses.keys())}  ✓")

    print("\n✅  PhysGTO-LPBF smoke test passed!")
