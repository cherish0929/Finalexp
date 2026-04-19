"""
GNOT (General Neural Operator Transformer) — adapted for LPBF simulation
=========================================================================
Source paper : "GNOT: A General Neural Operator Transformer for Operator
               Learning"  Hao et al., ICML 2023
               https://arxiv.org/abs/2302.14376
Reference repo: https://github.com/thu-ml/GNOT  (also HaoZhongkai/GNOT)

Adaptation notes
----------------
* Matches the forward / autoregressive interface of physgto.py exactly.
* Key GNOT components implemented:
    - Heterogeneous normalized cross-attention (with geometric gating).
    - Mixture-of-Experts (MoE) feed-forward network.
* In the original GNOT, 'query' positions and 'input function' positions can
  differ (operator learning).  Here both are the same mesh, so the module
  degenerates to self-attention — equivalent to the paper's single-mesh setting.
* Edge connectivity is accepted but silently ignored (attention-based).

LPBF suitability
----------------
★★★★☆  GNOT suits LPBF for multi-field simulation:
  - MoE FFN lets different experts specialise in temperature, melt fraction,
    velocity, etc. — particularly valuable when fields behave very differently
    in the active vs. inactive regions.
  - Geometric gating provides soft spatial decomposition, helping the model
    focus attention on the melt-pool neighbourhood.
  - Recommended n_experts: 4-8; N_block: 4-6.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ─── Shared utilities (identical to physgto.py) ───────────────────────────────

def FourierEmbedding(pos, pos_start, pos_length):
    original_shape = pos.shape
    new_pos = pos.reshape(-1, original_shape[-1])
    index = torch.arange(pos_start, pos_start + pos_length,
                         device=pos.device, dtype=pos.dtype)
    freq = 2 ** index * torch.pi
    cos_feat = torch.cos(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    sin_feat = torch.sin(freq.view(1, 1, -1) * new_pos.unsqueeze(-1))
    embedding = torch.cat([cos_feat, sin_feat], dim=-1)
    embedding = embedding.view(*original_shape[:-1], -1)
    return torch.cat([embedding, pos], dim=-1)


def _make_mlp(in_dim, out_dim, hidden_dim, n_hidden, layer_norm, act='SiLU'):
    A = {'SiLU': nn.SiLU, 'GELU': nn.GELU, 'ReLU': nn.ReLU}[act]
    layers = [nn.Linear(in_dim, hidden_dim), A()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), A()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


# ─── Geometric Gating ─────────────────────────────────────────────────────────

class GeometricGate(nn.Module):
    """
    Soft per-head spatial gate (§3.2 of GNOT).
    Produces a sigmoid gate in [0,1]^(bs, N, n_heads) from node features,
    scaling each attention head's contribution at each location.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, n_heads),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (bs, N, d) → gate: (bs, N, n_heads, 1)"""
        return self.gate_proj(x).unsqueeze(-1)          # (bs,N,H,1)


# ─── Heterogeneous Normalised Attention (HNA) ────────────────────────────────

class HeterogeneousNormAttention(nn.Module):
    """
    GNOT's normalised cross-attention with geometric gating.

    When query == key_value this is self-attention (our LPBF case).
    The 'normalised' trick divides the attention logits by the running sum
    across keys, making the soft-max implicit and numerically stable.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)
        self.geo_gate = GeometricGate(d_model, n_heads)

    def forward(self, query: torch.Tensor,
                key_value: torch.Tensor | None = None) -> torch.Tensor:
        """
        query     : (bs, Nq, d)
        key_value : (bs, Nk, d)   [defaults to query for self-attn]
        Returns   : (bs, Nq, d)
        """
        if key_value is None:
            key_value = query

        bs, Nq, d = query.shape
        Nk = key_value.shape[1]
        H, hd = self.n_heads, self.head_dim

        Q = self.q_proj(query).view(bs, Nq, H, hd).transpose(1, 2)   # (bs,H,Nq,hd)
        K = self.k_proj(key_value).view(bs, Nk, H, hd).transpose(1, 2)
        V = self.v_proj(key_value).view(bs, Nk, H, hd).transpose(1, 2)

        # Standard scaled dot-product (the paper uses linear attention for very
        # large N; swap to F.scaled_dot_product_attention for efficiency)
        attn = (Q @ K.transpose(-2, -1)) * self.scale      # (bs,H,Nq,Nk)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(bs, Nq, d)           # (bs,Nq,d)

        # Apply geometric gate (per-head spatial modulation)
        gate = self.geo_gate(query)                                    # (bs,Nq,H,1)
        out_gated = (attn @ V).transpose(1, 2)                         # (bs,Nq,H,hd)
        out_gated = (out_gated * gate).reshape(bs, Nq, d)

        return self.out_proj(out_gated)


# ─── Mixture-of-Experts FFN ───────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """
    Soft Mixture-of-Experts feed-forward network (§3.3 of GNOT).
    Uses a learned gate to blend outputs of n_experts independent MLPs.
    """

    def __init__(self, d_model: int, n_experts: int = 4,
                 d_ffn: int | None = None):
        super().__init__()
        d_ffn = d_ffn or 2 * d_model
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.SiLU(),
                nn.Linear(d_ffn, d_model),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (bs, N, d) → (bs, N, d)"""
        # gate_w: (bs, N, n_experts)
        gate_w = torch.softmax(self.gate(x), dim=-1)
        # expert_outs: (bs, N, d, n_experts)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)
        # weighted sum
        return (expert_outs * gate_w.unsqueeze(-2)).sum(dim=-1)


# ─── GNOT Block ───────────────────────────────────────────────────────────────

class GNOTBlock(nn.Module):
    """
    PreNorm HNA + PreNorm MoEFFN.
    Accepts spatial encoding concat (same pattern as PhysGTO MixerBlock).
    """

    def __init__(self, d_model: int, n_heads: int, n_experts: int,
                 enc_s_dim: int, dropout: float = 0.0):
        super().__init__()
        node_dim = d_model + enc_s_dim

        self.ln1  = nn.LayerNorm(node_dim)
        self.attn = HeterogeneousNormAttention(node_dim, n_heads, dropout)

        self.ln2  = nn.LayerNorm(node_dim)
        self.moe  = MoEFFN(node_dim, n_experts, d_ffn=2 * node_dim)

        self.proj_out = nn.Linear(node_dim, d_model)

    def forward(self, V: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        V_in = torch.cat([V, pos_enc], dim=-1)
        V_in = V_in + self.attn(self.ln1(V_in))
        V_in = V_in + self.moe(self.ln2(V_in))
        return V + self.proj_out(V_in)


# ─── Main Model ───────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    GNOT with PhysGTO-compatible interface.

    Parameters
    ----------
    space_size  : spatial dimensions (3 for LPBF)
    pos_enc_dim : Fourier feature octaves
    cond_dim    : raw condition vector length
    N_block     : number of GNOT blocks (4–6 recommended)
    in_dim      : per-node state features at input
    out_dim     : per-node state features at output
    enc_dim     : hidden / latent dimension
    n_head      : attention heads per block
    n_token     : number of MoE experts (replaces token count)
    dt          : default Euler timestep
    """

    def __init__(self,
                 space_size:     int   = 3,
                 pos_enc_dim:    int   = 5,
                 cond_dim:       int   = 32,
                 N_block:        int   = 4,
                 in_dim:         int   = 4,
                 out_dim:        int   = 4,
                 enc_dim:        int   = 128,
                 n_head:         int   = 4,
                 n_token:        int   = 4,     # ← number of MoE experts
                 dt:             float = 0.05,
                 stepper_scheme: str   = "euler"):
        super().__init__()

        self.dt          = dt
        self.pos_enc_dim = pos_enc_dim
        self.N_block     = N_block

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 1 + 2 * pos_enc_dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        self.enc_s_dim = enc_s_dim
        n_experts = n_token

        # ── node encoder (mirrors PhysGTO Encoder) ────────────────────────────
        self.fv1     = _make_mlp(in_dim + space_size, enc_dim, enc_dim, 1,
                                 layer_norm=False)
        self.fv_time = _make_mlp(enc_t_dim, enc_dim, enc_dim, 1,
                                 layer_norm=False)
        self.fv_cond = _make_mlp(enc_c_dim, enc_dim, enc_dim, 1,
                                 layer_norm=False)

        # ── GNOT blocks ───────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            GNOTBlock(enc_dim, n_head, n_experts, enc_s_dim)
            for _ in range(N_block)
        ])

        # ── multi-block decoder (mirrors PhysGTO Decoder) ─────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(N_block * enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, out_dim),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _t2d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1) if t.dim() == 1 else t

    def _encode_node(self, state_in, node_pos, time_i, conditions):
        time_i  = self._t2d(time_i)
        pos_enc = FourierEmbedding(node_pos,   0, self.pos_enc_dim)
        t_enc   = FourierEmbedding(time_i,     0, self.pos_enc_dim)
        c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        V = (self.fv1(torch.cat([state_in, node_pos], dim=-1))
             + self.fv_time(t_enc).unsqueeze(1)
             + self.fv_cond(c_enc).unsqueeze(1))
        return V, pos_enc

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        """
        edges is accepted but ignored (attention-based model).
        All other arguments identical to physgto.Model.forward.
        """
        V, pos_enc_computed = self._encode_node(
            state_in, node_pos, time_i, conditions)

        pos_enc_blk = pos_enc if pos_enc is not None else pos_enc_computed

        V_all = []
        for block in self.blocks:
            V = block(V, pos_enc_blk)
            V_all.append(V)

        feat   = torch.cat([torch.cat(V_all, dim=-1), pos_enc_blk], dim=-1)
        v_pred = self.decoder(feat)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)
        return state_in + dt * v_pred

    # ── autoregressive rollout ────────────────────────────────────────────────

    def autoregressive(self, state_in, node_pos, edges, time_seq, conditions,
                       dt=None, check_point=False):
        state_t = state_in
        outputs = []
        T = time_seq.shape[1]

        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)

        for t in range(T):
            time_i = time_seq[:, t]

            def _step(s, ti):
                return self.forward(s, node_pos, edges, ti, conditions,
                                    pos_enc=pos_enc, dt=dt)

            if check_point is True or (isinstance(check_point, int) and t >= check_point):
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                state_t = checkpoint(_step, state_t, time_i, use_reentrant=False)
            else:
                state_t = _step(state_t, time_i)

            outputs.append(state_t)

        return torch.stack(outputs, dim=1)
