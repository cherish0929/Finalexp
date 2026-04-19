"""
Transolver — adapted for LPBF physics simulation
=================================================
Source paper : "Transolver: A Fast Transformer Solver for PDEs on General
               Geometries"  Wu et al., ICML 2024 (Spotlight)
               https://arxiv.org/abs/2402.02366
Reference repo: https://github.com/thuml/Transolver

Adaptation notes
----------------
* Matches the forward / autoregressive interface of physgto.py exactly.
* The core Physics-Attention module replaces the GNN-based Mixer in PhysGTO.
* Time, condition, and position are encoded with the same FourierEmbedding.
* Edge connectivity is NOT used (Transolver is mesh-free / attention-based).
  The `edges` argument is accepted but silently ignored for compatibility.

LPBF suitability
----------------
★★★★★  Transolver is an excellent choice for LPBF:
  - Physics-Attention groups nodes by physical state (hot / cold / melting),
    which naturally separates the small active region from the cold bulk.
  - Global receptive field in O(N·M) — efficient even for large meshes.
  - Slice mechanism is rotation- and resolution-invariant (good for 3-D).
  - Recommended n_slices (M): 32–64 for typical LPBF domains.
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


# ─── Physics-Attention (core Transolver module) ───────────────────────────────

class PhysicsAttention(nn.Module):
    """
    Learns M 'physical-state slices'; mesh points vote into slices via a
    soft assignment, attention runs over the M slice tokens (O(M²) instead
    of O(N²)), and results are scattered back to every point.

    From §3.2 of the Transolver paper.
    """

    def __init__(self, d_model: int, n_heads: int, n_slices: int,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_slices = n_slices
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        # Soft slice-membership weights: each node → weight over M slices
        self.slice_proj = nn.Linear(d_model, n_slices)

        # Per-slice key / value encoding (from aggregated slice tokens)
        self.to_kv = nn.Linear(d_model, 2 * d_model, bias=False)

        # Per-point query encoding
        self.to_q  = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (bs, N, d_model)
        Returns (bs, N, d_model)
        """
        bs, N, d = x.shape
        M, H, hd = self.n_slices, self.n_heads, self.head_dim

        # ── 1. Soft slice assignment ──────────────────────────────────────────
        # w: (bs, N, M)  — normalised over the N dimension
        w_raw  = self.slice_proj(x)                          # (bs, N, M)
        w      = torch.softmax(w_raw, dim=-1)                # (bs, N, M)
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-6)    # normalize over N

        # ── 2. Aggregate nodes → slice tokens ─────────────────────────────────
        # tokens: (bs, M, d)
        tokens = torch.einsum('bnm,bnd->bmd', w_norm, x)

        # ── 3. Self-attention among slice tokens ──────────────────────────────
        # Queries: from each mesh point  (used later for scatter)
        # Keys/Values: from slice tokens
        kv = self.to_kv(tokens).reshape(bs, M, 2, H, hd)   # (bs,M,2,H,hd)
        k  = kv[:, :, 0].transpose(1, 2)                    # (bs,H,M,hd)
        v  = kv[:, :, 1].transpose(1, 2)                    # (bs,H,M,hd)
        q  = self.to_q(x).reshape(bs, N, H, hd).transpose(1, 2)  # (bs,H,N,hd)

        attn  = (q @ k.transpose(-2, -1)) * self.scale      # (bs,H,N,M)
        attn  = torch.softmax(attn, dim=-1)
        attn  = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bs, N, d)  # (bs,N,d)
        return self.out_proj(out)


# ─── Transolver Block ─────────────────────────────────────────────────────────

class TransolverBlock(nn.Module):
    """
    PreNorm  PhysicsAttention  +  PreNorm  FFN  with residual connections.
    """

    def __init__(self, d_model: int, n_heads: int, n_slices: int,
                 enc_s_dim: int, dropout: float = 0.0):
        super().__init__()

        # node features are concat with spatial encoding inside the block
        node_dim = d_model + enc_s_dim

        self.ln1  = nn.LayerNorm(node_dim)
        self.attn = PhysicsAttention(node_dim, n_heads, n_slices, dropout)

        self.ln2  = nn.LayerNorm(node_dim)
        self.ffn  = nn.Sequential(
            nn.Linear(node_dim, 2 * node_dim),
            nn.SiLU(),
            nn.Linear(2 * node_dim, node_dim),
        )
        # project back to d_model after block
        self.proj_out = nn.Linear(node_dim, d_model)

    def forward(self, V: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """
        V       : (bs, N, d_model)
        pos_enc : (bs, N, enc_s_dim)   — Fourier-encoded node positions
        """
        # concatenate spatial encoding at each layer (mirrors PhysGTO's MixerBlock)
        V_in = torch.cat([V, pos_enc], dim=-1)               # (bs,N,node_dim)
        V_in = V_in + self.attn(self.ln1(V_in))
        V_in = V_in + self.ffn(self.ln2(V_in))
        return V + self.proj_out(V_in)                       # residual onto V


# ─── Main Model ───────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    Transolver with PhysGTO-compatible interface.

    Parameters
    ----------
    space_size  : spatial dimensions (3 for LPBF)
    pos_enc_dim : Fourier feature octaves
    cond_dim    : raw condition vector length
    N_block     : number of Transolver blocks (4 as in paper)
    in_dim      : per-node state features at input
    out_dim     : per-node state features at output
    enc_dim     : hidden / latent dimension  (d_model)
    n_head      : attention heads per block
    n_token     : number of physics slices M (replaces token count)
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
                 n_token:        int   = 64,    # ← number of physics slices M
                 dt:             float = 0.05,
                 stepper_scheme: str   = "euler"):
        super().__init__()

        self.dt          = dt
        self.pos_enc_dim = pos_enc_dim

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size   # Fourier-pos
        enc_t_dim = 1 + 2 * pos_enc_dim                          # Fourier-time
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim             # Fourier-cond

        self.enc_s_dim = enc_s_dim
        n_slices = n_token

        # ── node encoder ──────────────────────────────────────────────────────
        node_in = in_dim + enc_s_dim + enc_t_dim + enc_c_dim
        # Following PhysGTO: fv1 handles state+pos, then add time and cond
        self.fv1     = _make_mlp(in_dim + space_size, enc_dim, enc_dim, 1,
                                 layer_norm=False)
        self.fv_time = _make_mlp(enc_t_dim, enc_dim, enc_dim, 1,
                                 layer_norm=False)
        self.fv_cond = _make_mlp(enc_c_dim, enc_dim, enc_dim, 1,
                                 layer_norm=False)

        # ── Transolver blocks ─────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransolverBlock(enc_dim, n_head, n_slices, enc_s_dim)
            for _ in range(N_block)
        ])

        # ── decoder ───────────────────────────────────────────────────────────
        # Collect all block outputs (same multi-block readout as PhysGTO)
        self.decoder = nn.Sequential(
            nn.Linear(N_block * enc_dim + enc_s_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, out_dim),
        )
        self.N_block = N_block

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _t2d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1) if t.dim() == 1 else t

    def _encode_node(self, state_in, node_pos, time_i, conditions):
        """Return V (bs, N, enc_dim) and pos_enc (bs, N, enc_s_dim)."""
        bs, N, _ = state_in.shape
        time_i = self._t2d(time_i)

        pos_enc = FourierEmbedding(node_pos,   0, self.pos_enc_dim)   # (bs,N,enc_s)
        t_enc   = FourierEmbedding(time_i,     0, self.pos_enc_dim)   # (bs,enc_t)
        c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)   # (bs,enc_c)

        # Mirrors PhysGTO Encoder: fv1 gets (state|pos), time/cond add to all nodes
        V = (self.fv1(torch.cat([state_in, node_pos], dim=-1))
             + self.fv_time(t_enc).unsqueeze(1)
             + self.fv_cond(c_enc).unsqueeze(1))
        return V, pos_enc

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        """
        edges is accepted but ignored (Transolver is attention-based).
        All other arguments are identical to physgto.Model.forward.
        """
        V, pos_enc_computed = self._encode_node(
            state_in, node_pos, time_i, conditions)

        # Use precomputed pos_enc if passed (for consistency with physgto API)
        if pos_enc is None:
            pos_enc_blk = pos_enc_computed
        else:
            pos_enc_blk = pos_enc   # (bs, N, enc_s_dim)

        V_all = []
        for block in self.blocks:
            V = block(V, pos_enc_blk)
            V_all.append(V)

        # Multi-block decoder (mirrors physgto Decoder)
        V_cat = torch.cat(V_all, dim=-1)                           # (bs,N,N_block*enc)
        feat  = torch.cat([V_cat, pos_enc_blk], dim=-1)
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

        # Precompute pos_enc (fixed) and c_enc (fixed) to avoid repeat work
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
