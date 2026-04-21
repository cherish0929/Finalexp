"""
MeshGraphNet (MGN) — adapted for LPBF physics simulation
=========================================================
Source paper : "Learning Mesh-Based Simulation with Graph Networks"
               Pfaff et al., ICLR 2021  https://arxiv.org/abs/2010.03409
Reference repo: https://github.com/echowve/meshGraphNets_pytorch

Adaptation notes
----------------
* Matches the forward / autoregressive interface of physgto.py exactly.
* time_i and conditions are encoded with FourierEmbedding, then broadcast to
  every node — exactly as in PhysGTO's Encoder.
* Euler integration step (state_in + dt * v_pred) is kept identical.
* n_head / n_token are accepted but unused (kept for drop-in compatibility).

LPBF suitability
----------------
★★★★☆  MGN is a natural fit for LPBF:
  - Graph structure captures local heat conduction / diffusion along edges.
  - Multiple GNN layers widen the receptive field without attention overhead.
  - Works well when the active (melt-pool) region is a small fraction of nodes.
  - Recommended N_block: 12-15 for full accuracy; 6-8 for faster iteration.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint


# ─── Shared utilities (identical to physgto.py) ───────────────────────────────

def FourierEmbedding(pos, pos_start, pos_length):
    """Concatenate [cos(2^i π x), sin(2^i π x), x] Fourier features."""
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


def get_edge_info(edges, node_pos):
    """Compute [d, -d, ‖d‖] edge features from node positions.
    Handles -1 padding by clamping to [0, N-1]; callers zero-out padded rows."""
    N = node_pos.shape[-2]
    safe = edges.clamp(min=0, max=N - 1)
    s = torch.gather(node_pos, -2, safe[..., 0:1].expand(-1, -1, node_pos.shape[-1]))
    r = torch.gather(node_pos, -2, safe[..., 1:2].expand(-1, -1, node_pos.shape[-1]))
    d    = r - s
    norm = d.norm(dim=-1, keepdim=True)
    return torch.cat([d, -d, norm], dim=-1)            # (bs, ne, 2·space+1)


def _make_mlp(in_dim, out_dim, hidden_dim, n_hidden, layer_norm, act='SiLU'):
    A = {'SiLU': nn.SiLU, 'GELU': nn.GELU, 'ReLU': nn.ReLU}[act]
    layers = [nn.Linear(in_dim, hidden_dim), A()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), A()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


# ─── MGN Processor Block ──────────────────────────────────────────────────────

class MGNBlock(nn.Module):
    """One round of edge-then-node message passing with residual connections."""

    def __init__(self, node_dim: int, edge_dim: int, n_hidden: int = 2):
        super().__init__()
        self.f_edge = _make_mlp(2 * node_dim + edge_dim, edge_dim,
                                edge_dim, n_hidden, layer_norm=True)
        self.f_node = _make_mlp(node_dim + edge_dim, node_dim,
                                node_dim, n_hidden, layer_norm=True)

    def forward(self, V, E, edges):
        bs, N, _ = V.shape

        # valid_mask: (bs, ne) — False for -1-padded rows
        valid_mask = (edges >= 0).all(-1)  # (bs, ne)

        safe = edges.clamp(min=0, max=N - 1)
        v_s = torch.gather(V, 1, safe[..., 0:1].expand(-1, -1, V.shape[-1]))   # sender features
        v_r = torch.gather(V, 1, safe[..., 1:2].expand(-1, -1, V.shape[-1]))   # receiver features

        e_delta = self.f_edge(torch.cat([v_s, v_r, E], dim=-1))
        e_delta = e_delta * valid_mask.unsqueeze(-1)    # zero-out padded rows
        E = E + e_delta                                  # edge residual

        recv_idx = safe[..., 1:2].expand(-1, -1, E.shape[-1])
        agg = scatter_mean(E * valid_mask.unsqueeze(-1), recv_idx, dim=1, dim_size=N)
        v_delta = self.f_node(torch.cat([V, agg], dim=-1))
        V = V + v_delta                                  # node residual
        return V, E


# ─── Main Model ───────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    MeshGraphNet with PhysGTO-compatible interface.

    Parameters (same defaults as PhysGTO for drop-in comparison)
    ------------------------------------------------------------
    space_size  : spatial dimensions (3 for LPBF)
    pos_enc_dim : Fourier feature octaves for position / time / condition
    cond_dim    : raw condition vector length
    N_block     : number of GNN message-passing layers (12–15 recommended)
    in_dim      : per-node state features at input
    out_dim     : per-node state features at output
    enc_dim     : hidden / latent dimension
    n_head      : accepted but unused (interface compat only)
    n_token     : accepted but unused (interface compat only)
    dt          : default Euler timestep
    """

    def __init__(self,
                 space_size:     int   = 3,
                 pos_enc_dim:    int   = 5,
                 cond_dim:       int   = 32,
                 N_block:        int   = 12,
                 in_dim:         int   = 4,
                 out_dim:        int   = 4,
                 enc_dim:        int   = 128,
                 n_head:         int   = 4,     # unused
                 n_token:        int   = 128,   # unused
                 dt:             float = 0.05,
                 stepper_scheme: str   = "euler"):
        super().__init__()

        self.dt          = dt
        self.pos_enc_dim = pos_enc_dim

        # ── dimension accounting (mirrors physgto.py) ─────────────────────────
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size   # Fourier-pos dim
        enc_t_dim = 1 + 2 * pos_enc_dim                          # Fourier-time dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim             # Fourier-cond dim
        edge_raw  = 2 * space_size + 1                           # [d, -d, ‖d‖]

        node_in = in_dim + enc_s_dim + enc_t_dim + enc_c_dim

        # ── encoder ───────────────────────────────────────────────────────────
        self.node_enc = _make_mlp(node_in,  enc_dim, enc_dim, 1, layer_norm=True)
        self.edge_enc = _make_mlp(edge_raw, enc_dim, enc_dim, 1, layer_norm=True)

        # ── processor ─────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            MGNBlock(enc_dim, enc_dim) for _ in range(N_block)
        ])

        # ── decoder ───────────────────────────────────────────────────────────
        self.decoder = _make_mlp(enc_dim, out_dim, enc_dim, 2, layer_norm=False)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _t2d(t: torch.Tensor) -> torch.Tensor:
        """Guarantee time tensor is (bs, 1) for FourierEmbedding."""
        return t.unsqueeze(-1) if t.dim() == 1 else t

    def _encode(self, state_in, node_pos, edges, time_i, conditions):
        bs, N, _ = state_in.shape
        time_i = self._t2d(time_i)

        pos_enc = FourierEmbedding(node_pos,   0, self.pos_enc_dim)   # (bs,N,enc_s)
        t_enc   = FourierEmbedding(time_i,     0, self.pos_enc_dim)   # (bs,enc_t)
        c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)   # (bs,enc_c)

        t_enc = t_enc.unsqueeze(1).expand(-1, N, -1)       # (bs,N,enc_t)
        c_enc = c_enc.unsqueeze(1).expand(-1, N, -1)       # (bs,N,enc_c)

        node_feat = torch.cat([state_in, pos_enc, t_enc, c_enc], dim=-1)
        V = self.node_enc(node_feat)
        E = self.edge_enc(get_edge_info(edges, node_pos))
        return V, E

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        """
        Inputs  (same as physgto.Model.forward)
        ----------------------------------------
        state_in   : (bs, N, in_dim)
        node_pos   : (bs, N, space_size)
        edges      : (bs, ne, 2)  — long or int, converted internally
        time_i     : (bs, 1)  or  (bs,)
        conditions : (bs, cond_dim)
        pos_enc, c_enc : precomputed Fourier embeddings (optional, ignored here)
        dt         : override default timestep

        Returns
        -------
        state_pred : (bs, N, out_dim)
        """
        edges = edges.long() if edges.dtype != torch.long else edges
        V, E  = self._encode(state_in, node_pos, edges, time_i, conditions)

        for block in self.blocks:
            V, E = block(V, E, edges)

        v_pred = self.decoder(V)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)
        return state_in + dt * v_pred

    # ── autoregressive rollout ────────────────────────────────────────────────

    def autoregressive(self, state_in, node_pos, edges, time_seq, conditions,
                       dt=None, check_point=False):
        """
        Inputs  (same as physgto.Model.autoregressive)
        -----------------------------------------------
        state_in   : (bs, N, in_dim)
        time_seq   : (bs, T)  or  (bs, T, 1)
        conditions : (bs, cond_dim)

        Returns
        -------
        outputs : (bs, T, N, out_dim)
        """
        state_t = state_in
        outputs = []
        T = time_seq.shape[1]

        for t in range(T):
            time_i = time_seq[:, t]

            def _step(s, ti):
                return self.forward(s, node_pos, edges, ti, conditions, dt=dt)

            if check_point is True or (isinstance(check_point, int) and t >= check_point):
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                state_t = checkpoint(_step, state_t, time_i, use_reentrant=False)
            else:
                state_t = _step(state_t, time_i)

            outputs.append(state_t)

        return torch.stack(outputs, dim=1)   # (bs, T, N, out_dim)
