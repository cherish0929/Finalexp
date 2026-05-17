"""
physgto_lpbf_v2.py — PhysGTO-LPBF v2: Source-Term Mixer Architecture

Architecture:
  1. LaserFieldModule + ScaleAwareEncoder  (reused from physgto_lpbf.py)
  2. MultiFieldMixer                       (reused from physgto_attnres_multi_v3.py)
  3. SourceTermProjector  — compress N nodes → n_src tokens × 5 source terms
       A. DiffusionTerm   (GNN neighbor differencing)
       B. ConvectionTerm  (signed directional edge gradients)
       C. SourceSinkTerm  (laser + field state + thermal bias fusion)
       D. SurfaceTerm     (interface/boundary features with α(1-α) indicator)
       E. ResidualTerm    (learned projection)
  4. SourceTermMixer      — pure attention blocks with SwiGLU FFN, stochastic depth,
                            Block AttnRes, cross-source attention
  5. SourceBroadcastDecoder — multi-layer attention-based fusion: node queries attend
                              to all source tokens → FFN → per-field correction

Registered as model name "gto_lpbf_v2".
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

from src.model.physgto_lpbf import (
    LaserFieldModule,
    ScaleAwareEncoder,
    DEFAULT_PHYSICS_PARAMS,
    _encode_time_lpbf,
)


def _trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


# =============================================================================
# SwiGLU FFN — Gated Linear Unit variant (LLaMA / PaLM style)
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU(x) = (Wx ⊙ SiLU(Vx)) W_out — more expressive than standard FFN."""

    def __init__(self, dim: int, hidden_mult: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        hidden = ((hidden + 7) // 8) * 8
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_val = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_val(x)))


# =============================================================================
# Source-Term Feature Extractors (per-node, before compression)
# =============================================================================

class DiffusionExtractor(nn.Module):
    """GNN neighbor differencing — captures Laplacian-like local structure."""

    def __init__(self, enc_dim: int, d_src: int, edge_raw_dim: int):
        super().__init__()
        self.msg_mlp = MLP(input_size=enc_dim + edge_raw_dim, output_size=d_src,
                           hidden_size=d_src, n_hidden=1, act='SiLU', layer_norm=True)
        self.node_mlp = MLP(input_size=enc_dim + d_src, output_size=d_src,
                            hidden_size=d_src, n_hidden=1, act='SiLU', layer_norm=True)

    def forward(self, V, edges, edge_raw):
        bs, N, D = V.shape
        s_idx = edges[..., 0]
        r_idx = edges[..., 1]

        idx_s = s_idx.unsqueeze(-1).expand(-1, -1, D)
        idx_r = r_idx.unsqueeze(-1).expand(-1, -1, D)
        diff = torch.gather(V, 1, idx_r) - torch.gather(V, 1, idx_s)

        msg = self.msg_mlp(torch.cat([diff, edge_raw], dim=-1))
        col = r_idx.unsqueeze(-1).expand_as(msg)
        agg = scatter_add(msg, col, dim=1, dim_size=N)

        return self.node_mlp(torch.cat([V, agg], dim=-1))


class ConvectionExtractor(nn.Module):
    """Directional edge gradients with signed displacement — convective transport."""

    def __init__(self, enc_dim: int, d_src: int, edge_raw_dim: int):
        super().__init__()
        self.edge_mlp = MLP(input_size=enc_dim * 2 + edge_raw_dim, output_size=d_src,
                            hidden_size=d_src, n_hidden=1, act='SiLU', layer_norm=True)
        self.attn_net = nn.Sequential(
            nn.Linear(d_src, d_src // 4), nn.SiLU(), nn.Linear(d_src // 4, 1)
        )
        self.node_mlp = MLP(input_size=enc_dim + d_src, output_size=d_src,
                            hidden_size=d_src, n_hidden=1, act='SiLU', layer_norm=True)

    def forward(self, V, edges, edge_raw):
        bs, N, D = V.shape
        s_idx = edges[..., 0].unsqueeze(-1).expand(-1, -1, D)
        r_idx_flat = edges[..., 1]

        V_s = torch.gather(V, 1, s_idx)
        V_r = torch.gather(V, 1, r_idx_flat.unsqueeze(-1).expand(-1, -1, D))
        edge_feat = self.edge_mlp(torch.cat([V_s, V_r - V_s, edge_raw], dim=-1))
        del V_s, V_r

        logits = self.attn_net(edge_feat).squeeze(-1).clamp(-30, 30)
        alpha = scatter_softmax(logits, r_idx_flat, dim=1)
        col = r_idx_flat.unsqueeze(-1).expand_as(edge_feat)
        agg = scatter_add(alpha.unsqueeze(-1) * edge_feat, col, dim=1, dim_size=N)

        return self.node_mlp(torch.cat([V, agg], dim=-1))


class SourceSinkExtractor(nn.Module):
    """Laser energy source + thermal state bias → source/sink features."""

    def __init__(self, enc_dim: int, d_src: int, d_laser: int, n_fields: int):
        super().__init__()
        in_dim = enc_dim + d_laser + n_fields
        self.gate_proj = nn.Linear(d_laser, enc_dim)
        self.mlp = MLP(input_size=in_dim, output_size=d_src,
                       hidden_size=d_src, n_hidden=2, act='SiLU', layer_norm=True)

    def forward(self, V, laser_feat, state_in):
        gate = torch.sigmoid(self.gate_proj(laser_feat))
        V_gated = V * gate
        return self.mlp(torch.cat([V_gated, laser_feat, state_in], dim=-1))


class SurfaceExtractor(nn.Module):
    """Interface/boundary features with α(1-α) indicator for melt pool boundary."""

    def __init__(self, enc_dim: int, d_src: int, n_fields: int, alpha_idx: Optional[int]):
        super().__init__()
        self.alpha_idx = alpha_idx
        extra = 1 if alpha_idx is not None else 0
        in_dim = enc_dim + n_fields + extra
        self.mlp = MLP(input_size=in_dim, output_size=d_src,
                       hidden_size=d_src, n_hidden=2, act='SiLU', layer_norm=True)

    def forward(self, V, state_in):
        parts = [V, state_in]
        if self.alpha_idx is not None:
            alpha = state_in[..., self.alpha_idx:self.alpha_idx + 1]
            parts.append(4.0 * alpha * (1.0 - alpha))
        return self.mlp(torch.cat(parts, dim=-1))


class ResidualExtractor(nn.Module):
    """Learned projection capturing unmodelled physics."""

    def __init__(self, enc_dim: int, d_src: int):
        super().__init__()
        self.mlp = MLP(input_size=enc_dim, output_size=d_src,
                       hidden_size=d_src, n_hidden=2, act='SiLU', layer_norm=True)

    def forward(self, V):
        return self.mlp(V)


# =============================================================================
# SourceTermProjector: N nodes → n_src tokens per source term
# =============================================================================

class SourceTermProjector(nn.Module):
    """
    For each of 5 source terms, extract per-node features and compress
    to n_src tokens via cross-attention with learned queries.
    """

    def __init__(
        self,
        n_source_terms: int,
        n_src_tokens: int,
        enc_dim: int,
        d_src: int,
        d_laser: int,
        n_fields: int,
        space_size: int = 3,
        n_heads: int = 4,
        alpha_idx: Optional[int] = None,
    ):
        super().__init__()
        self.n_source_terms = n_source_terms
        self.n_src_tokens = n_src_tokens
        edge_raw_dim = 2 * space_size + 1

        self.extractors = nn.ModuleList([
            DiffusionExtractor(enc_dim, d_src, edge_raw_dim),
            ConvectionExtractor(enc_dim, d_src, edge_raw_dim),
            SourceSinkExtractor(enc_dim, d_src, d_laser, n_fields),
            SurfaceExtractor(enc_dim, d_src, n_fields, alpha_idx),
            ResidualExtractor(enc_dim, d_src),
        ])

        self.queries = nn.ParameterList([
            nn.Parameter(torch.empty(n_src_tokens, d_src))
            for _ in range(n_source_terms)
        ])
        for q in self.queries:
            _trunc_normal_(q, std=0.02)

        self.compress_attns = nn.ModuleList([
            nn.MultiheadAttention(d_src, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_source_terms)
        ])

        self.ln_feats = nn.ModuleList([
            nn.LayerNorm(d_src) for _ in range(n_source_terms)
        ])
        self.ln_queries = nn.ModuleList([
            nn.LayerNorm(d_src) for _ in range(n_source_terms)
        ])

    def forward(self, V_last, laser_feat, state_in, edges, node_pos):
        edge_raw = get_edge_info(edges, node_pos)
        tokens = []

        for k in range(self.n_source_terms):
            if k <= 1:
                feat = self.extractors[k](V_last, edges, edge_raw)
            elif k == 2:
                feat = self.extractors[k](V_last, laser_feat, state_in)
            elif k == 3:
                feat = self.extractors[k](V_last, state_in)
            else:
                feat = self.extractors[k](V_last)

            feat = self.ln_feats[k](feat)
            bs = feat.shape[0]
            Q = self.ln_queries[k](self.queries[k].unsqueeze(0).expand(bs, -1, -1))
            S_k, _ = self.compress_attns[k](Q, feat, feat)
            tokens.append(S_k)

        return tokens


# =============================================================================
# SourceMixerBlock: pure attention with SwiGLU, stochastic depth, Block AttnRes
# =============================================================================

class SourceMixerBlock(nn.Module):
    """
    Per source-term:
      1. Block AttnRes → Self-Attention (Atten)
      2. Cross-Source Attention (FieldCrossAttention)
      3. Block AttnRes → SwiGLU FFN
    Supports stochastic depth (drop entire block at configurable rate).
    """

    def __init__(self, d_src: int, n_source_terms: int, n_heads: int = 4,
                 n_token: int = 64, cross_heads: int = 4, n_latent: int = 2,
                 ffn_dropout: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.n_source_terms = n_source_terms
        self.drop_path_rate = drop_path_rate

        self.ln1s = nn.ModuleList([nn.LayerNorm(d_src) for _ in range(n_source_terms)])
        self.self_attns = nn.ModuleList([
            Atten(n_token=n_token, c_dim=d_src, n_heads=n_heads, n_latent=n_latent)
            for _ in range(n_source_terms)
        ])

        if n_source_terms > 1:
            self.cross_attns = nn.ModuleList([
                FieldCrossAttention(d_src, n_heads=cross_heads, n_token=n_token)
                for _ in range(n_source_terms)
            ])
        else:
            self.cross_attns = None

        self.ln2s = nn.ModuleList([nn.LayerNorm(d_src) for _ in range(n_source_terms)])
        self.ffns = nn.ModuleList([
            SwiGLU(d_src, hidden_mult=8 / 3, dropout=ffn_dropout)
            for _ in range(n_source_terms)
        ])

        self.attn_res_w = nn.ParameterList([
            nn.Parameter(torch.zeros(d_src))
            for _ in range(n_source_terms * 2)
        ])
        self.attn_res_norm = RMSNorm()

    def _drop_path(self, x, residual):
        if not self.training or self.drop_path_rate == 0.0:
            return residual + x
        keep = torch.rand(1, device=x.device).item() > self.drop_path_rate
        if keep:
            return residual + x / (1.0 - self.drop_path_rate)
        return residual

    def forward(self, S_list, blocks_list):
        norm = self.attn_res_norm
        S_out = []

        for k in range(self.n_source_terms):
            w_attn = self.attn_res_w[k * 2]
            h = block_attn_res(blocks_list[k], S_list[k], w_attn, norm)
            attn_out = self.self_attns[k](self.ln1s[k](h))
            S_out.append(self._drop_path(attn_out, S_list[k]))

        if self.n_source_terms > 1:
            S_cross = []
            for k in range(self.n_source_terms):
                others = [S_out[j] for j in range(self.n_source_terms) if j != k]
                other_cat = torch.cat(others, dim=1) if len(others) > 1 else others[0]
                cross_info = self.cross_attns[k](S_out[k], other_cat)
                S_cross.append(S_out[k] + cross_info)
        else:
            S_cross = S_out

        S_final = []
        for k in range(self.n_source_terms):
            w_ffn = self.attn_res_w[k * 2 + 1]
            h = block_attn_res(blocks_list[k], S_cross[k], w_ffn, norm)
            ffn_out = self.ffns[k](self.ln2s[k](h))
            S_final.append(self._drop_path(ffn_out, S_cross[k]))

        for k in range(self.n_source_terms):
            blocks_list[k] = blocks_list[k] + [S_final[k]]

        return S_final, blocks_list


# =============================================================================
# SourceTermMixer: stack of SourceMixerBlocks with linearly increasing drop path
# =============================================================================

class SourceTermMixer(nn.Module):

    def __init__(self, N_src_block: int, d_src: int, n_source_terms: int,
                 n_heads: int = 4, n_token: int = 64, cross_heads: int = 4,
                 n_latent: int = 2, ffn_dropout: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.n_source_terms = n_source_terms
        dpr = [drop_path_rate * i / max(N_src_block - 1, 1) for i in range(N_src_block)]
        self.blocks = nn.ModuleList([
            SourceMixerBlock(d_src, n_source_terms, n_heads, n_token,
                             cross_heads, n_latent, ffn_dropout, dpr[i])
            for i in range(N_src_block)
        ])

    def forward(self, S_tokens):
        blocks_list = [[S_tokens[k]] for k in range(self.n_source_terms)]
        S_list = S_tokens

        for block in self.blocks:
            S_list, blocks_list = block(S_list, blocks_list)

        return S_list


# =============================================================================
# SourceBroadcastDecoder: multi-layer attention-based fusion back to N nodes
# =============================================================================

class SourceBroadcastDecoder(nn.Module):
    """
    Fuses all refined source tokens back to N nodes via attention.
    Query = f(pos_enc, state_in, V_agg, laser_feat)
    Key/Value = concatenated source tokens from all terms.

    Two-stage: cross-attention → residual SwiGLU → output projection.
    Per-field learnable correction scales (initialized small).
    """

    def __init__(self, d_src: int, enc_dim: int, enc_s_dim: int,
                 d_laser: int, n_fields: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_fields = n_fields
        query_in_dim = enc_s_dim + n_fields + enc_dim + d_laser
        self.query_proj = MLP(input_size=query_in_dim, output_size=d_src,
                              hidden_size=d_src, n_hidden=1, act='SiLU', layer_norm=True)

        self.ln_q = nn.LayerNorm(d_src)
        self.ln_kv = nn.LayerNorm(d_src)
        self.cross_attn = nn.MultiheadAttention(d_src, n_heads, dropout=dropout,
                                                batch_first=True)

        self.ln_ffn = nn.LayerNorm(d_src)
        self.ffn = SwiGLU(d_src, hidden_mult=8 / 3, dropout=dropout)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_src),
            nn.Linear(d_src, n_fields),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

        self.correction_scale = nn.Parameter(torch.full((n_fields,), 0.1))

    def forward(self, S_refined_list, pos_enc, state_in, V_agg, laser_feat):
        S_all = torch.cat(S_refined_list, dim=1)

        query_input = torch.cat([pos_enc, state_in, V_agg, laser_feat], dim=-1)
        Q = self.query_proj(query_input)

        Q_n = self.ln_q(Q)
        KV = self.ln_kv(S_all)
        attn_out, _ = self.cross_attn(Q_n, KV, KV)
        h = Q + attn_out

        h = h + self.ffn(self.ln_ffn(h))

        raw = self.out_proj(h)
        return raw * self.correction_scale.unsqueeze(0).unsqueeze(0)


# =============================================================================
# Full Model
# =============================================================================

class Model(nn.Module):
    """
    PhysGTO-LPBF v2 (model name: "gto_lpbf_v2")

    Source-Term Mixer architecture with learnable source-term token compression,
    attention-based mixing, and attention-based broadcast back to nodes.
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
        # Source-term mixer parameters
        n_source_terms: int = 5,
        n_src_tokens: int = 256,
        d_src: int = 64,
        N_src_block: int = 2,
        src_n_heads: int = 4,
        src_cross_heads: int = 4,
        src_n_token: int = 64,
        src_n_latent: int = 2,
        # Aux loss weights
        ortho_weight: float = 0.01,
        balance_weight: float = 0.03,
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
        self.n_source_terms = n_source_terms
        self.ortho_weight = ortho_weight
        self.balance_weight = balance_weight
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

        # === Multi-Field Mixer (from v3) ===
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

        # === Source-Term Projector ===
        alpha_idx = fields.index("alpha.air") if self.has_alpha else None
        self.src_projector = SourceTermProjector(
            n_source_terms=n_source_terms,
            n_src_tokens=n_src_tokens,
            enc_dim=enc_dim,
            d_src=d_src,
            d_laser=d_laser,
            n_fields=self.n_fields,
            space_size=space_size,
            n_heads=src_n_heads,
            alpha_idx=alpha_idx,
        )

        # === Source-Term Mixer ===
        self.src_mixer = SourceTermMixer(
            N_src_block=N_src_block,
            d_src=d_src,
            n_source_terms=n_source_terms,
            n_heads=src_n_heads,
            n_token=src_n_token,
            cross_heads=src_cross_heads,
            n_latent=src_n_latent,
            ffn_dropout=0.1,
            drop_path_rate=0.1,
        )

        # === Source Broadcast Decoder ===
        self.src_decoder = SourceBroadcastDecoder(
            d_src=d_src,
            enc_dim=enc_dim,
            enc_s_dim=enc_s_dim,
            d_laser=d_laser,
            n_fields=self.n_fields,
            n_heads=src_cross_heads,
            dropout=0.1,
        )

        self.aux_losses = {}

        # Bounded output controls
        self.T_idx = fields.index("T") if "T" in fields else None
        self.alpha_idx = fields.index("alpha.air") if "alpha.air" in fields else None
        self.gamma_idx = fields.index("gamma_liquid") if "gamma_liquid" in fields else None
        self.max_delta_T = 2.0

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
            _, laser_feat = self.laser_module(
                node_pos_abs, laser_pos, laser_params, alpha_for_laser
            )
        else:
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

        # Multi-Field Mixer
        V_all_list = self.mixer(V_list, E, edges_long, pos_enc_out)

        # Base Decoder
        v_pred = self.decoder(V_all_list, pos_enc_out)

        # Source-Term Processing
        n_f = len(V_all_list)
        V_last = sum(V_all_list[f][:, -1] for f in range(n_f)) / n_f
        S_tokens = self.src_projector(V_last, laser_feat, state_in, edges_long, node_pos)
        S_refined = self.src_mixer(S_tokens)

        src_correction = self.src_decoder(
            S_refined, pos_enc_out, state_in, V_last, laser_feat
        )

        combined = v_pred + src_correction

        # Aux losses
        self._compute_aux_losses(S_refined)

        # Time stepping
        if self.stepper_scheme == "euler":
            with autocast(device_type="cuda", enabled=False):
                delta = combined.float() * dt_tensor.unsqueeze(-1).float()
                if self.T_idx is not None:
                    bounded_T = torch.tanh(
                        delta[..., self.T_idx:self.T_idx + 1] / self.max_delta_T
                    ) * self.max_delta_T
                    delta = torch.cat([
                        delta[..., :self.T_idx],
                        bounded_T,
                        delta[..., self.T_idx + 1:],
                    ], dim=-1)
                state_pred = state_in.float() + delta
        else:
            delta = combined
            if self.T_idx is not None:
                bounded_T = torch.tanh(
                    delta[..., self.T_idx:self.T_idx + 1] / self.max_delta_T
                ) * self.max_delta_T
                delta = torch.cat([
                    delta[..., :self.T_idx],
                    bounded_T,
                    delta[..., self.T_idx + 1:],
                ], dim=-1)
            state_pred = state_in + delta

        # Clamp VoF fields
        channels = list(state_pred.unbind(dim=-1))
        if self.alpha_idx is not None:
            channels[self.alpha_idx] = channels[self.alpha_idx].clamp(0.0, 1.0)
        if self.gamma_idx is not None:
            channels[self.gamma_idx] = channels[self.gamma_idx].clamp(0.0, 1.0)
        state_pred = torch.stack(channels, dim=-1)

        return state_pred

    def _compute_aux_losses(self, S_refined):
        """Source-term diversity (orthogonality) + token utilization (balance) losses."""
        eps = 1e-6

        # Orthogonality: encourage different source terms to encode different information
        means = torch.stack([S_k.mean(dim=1) for S_k in S_refined], dim=1)
        normed = means / (means.norm(dim=-1, keepdim=True) + eps)
        gram = torch.bmm(normed, normed.transpose(1, 2))
        eye = torch.eye(len(S_refined), device=gram.device).unsqueeze(0)
        ortho_loss = ((gram - eye) ** 2).mean()

        # Balance: maximize entropy of per-token attention norms → prevent token collapse
        all_tokens = torch.cat(S_refined, dim=1)
        token_norms = all_tokens.norm(dim=-1)
        token_probs = F.softmax(token_norms, dim=-1)
        entropy = -(token_probs * torch.log(token_probs + eps)).sum(dim=-1).mean()
        max_entropy = math.log(all_tokens.shape[1])
        balance_loss = 1.0 - entropy / max_entropy

        self.aux_losses = {
            "ortho": (ortho_loss, self.ortho_weight),
            "balance": (balance_loss, self.balance_weight),
        }

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

            laser_pos_t = laser_traj[:, t + 1] if laser_traj is not None else None
            abs_time_t = abs_time_seq[:, t + 1] if abs_time_seq is not None else None
            alpha_idx = self.fields.index("alpha.air") if self.has_alpha else None
            alpha_in_t = state_t[..., alpha_idx:alpha_idx + 1].clone() if alpha_idx is not None else None

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
# Smoke test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print("PhysGTO-LPBF v2 smoke test")
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
            n_source_terms=5,
            n_src_tokens=16,
            d_src=32,
            N_src_block=2,
            src_n_heads=4,
            src_cross_heads=4,
            src_n_token=16,
            src_n_latent=2,
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
        laser_traj = torch.zeros(bs, T + 1, 3)
        abs_time_seq = torch.linspace(0, 4e-4, T + 1).unsqueeze(0).expand(bs, -1)

        out = model.autoregressive(
            state_in, node_pos, edges, time_seq,
            spatial_inform, conditions, dt=2e-5,
            node_pos_abs=node_pos_abs, laser_params=laser_params,
            laser_traj=laser_traj, abs_time_seq=abs_time_seq,
        )
        assert out.shape == (bs, T, N, n_f), f"Shape mismatch: {out.shape}"
        print(f"  fields={test_fields}  out={out.shape}  "
              f"aux_losses={list(model.aux_losses.keys())}  ✓")

    # Test backward
    model_bw = Model(
        fields=["T", "alpha.air"], space_size=3, pos_enc_dim=3, cond_dim=8,
        spatial_dim=10, N_block=2, in_dim=2, out_dim=2, enc_dim=64,
        n_head=4, n_token=32, n_latent=2, n_fields=2, d_laser=16, dt=2e-5,
        n_source_terms=5, n_src_tokens=16, d_src=32, N_src_block=2,
        src_n_heads=4, src_cross_heads=4, src_n_token=16, src_n_latent=2,
    )
    model_bw.train()
    out_bw = model_bw.autoregressive(
        torch.randn(2, 64, 2), torch.rand(2, 64, 3),
        torch.randint(0, 64, (2, 128, 2)),
        torch.linspace(0, 4e-4, 4).unsqueeze(0).expand(2, -1),
        spatial_inform, torch.randn(2, 8), dt=2e-5,
        node_pos_abs=torch.rand(2, 64, 3) * 1e-3,
        laser_params=laser_params, laser_traj=torch.zeros(2, 5, 3),
        abs_time_seq=torch.linspace(0, 4e-4, 5).unsqueeze(0).expand(2, -1),
    )
    loss = out_bw.sum()
    loss.backward()
    print("  backward pass ✓")

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  params: {params / 1e6:.3f}M")
    print("\n✅  PhysGTO-LPBF v2 smoke test passed!")
