"""
baseline_models.py
==================
Three operator-learning baselines for LPBF physical field comparison.
All models expose EXACTLY the same interface as physgto.py:

    forward(state_in, node_pos, edges, time_i, conditions,
            pos_enc=None, c_enc=None, dt=None) -> state_pred

    autoregressive(state_in, node_pos, edges, time_seq, conditions,
                   dt=None, check_point=False) -> outputs  # (bs, T, N, out_dim)

Models
------
1. GraphViTModel   – GNN + Transformer  (任意图，无结构假设)
2. FNO3DModel      – 3D Fourier Neural Operator  (均匀立方体网格)
3. UNet3DModel     – 3D U-Net  (均匀立方体网格)

Grid-based models (FNO3D, UNet3D) require a `grid_shape=(D, H, W)` argument
where D*H*W == N (number of nodes). The node ordering in state_in / node_pos
must follow C-order (Z-major, i.e. numpy default reshape order).

Dependencies
------------
    torch, torch_scatter          # same as physgto.py
    (No extra packages required)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint

# ===========================================================================
# Shared utilities  (identical to physgto.py)
# ===========================================================================

def FourierEmbedding(pos, pos_start, pos_length):
    """Fourier positional embedding: F(x) = [cos(2^i*pi*x), sin(2^i*pi*x), x]"""
    original_shape = pos.shape
    p = pos.reshape(-1, original_shape[-1])
    idx = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq = (2 ** idx) * torch.pi                          # (pos_length,)
    cos_f = torch.cos(freq[None, None, :] * p.unsqueeze(-1))  # (B*N, D, L)
    sin_f = torch.sin(freq[None, None, :] * p.unsqueeze(-1))
    emb = torch.cat([cos_f, sin_f], dim=-1).reshape(*original_shape[:-1], -1)
    return torch.cat([emb, pos], dim=-1)


def get_edge_info(edges, node_pos):
    """Compute relative displacement vectors for each edge.
    Handles -1 padding by clamping to [0, N-1]; callers zero-out padded rows."""
    N = node_pos.shape[-2]
    safe = edges.clamp(min=0, max=N - 1)
    idx0 = safe[..., 0:1].expand(-1, -1, node_pos.shape[-1])
    idx1 = safe[..., 1:2].expand(-1, -1, node_pos.shape[-1])
    s = torch.gather(node_pos, -2, idx0)
    r = torch.gather(node_pos, -2, idx1)
    d = r - s
    norm = torch.sqrt((d ** 2).sum(-1, keepdim=True))
    return torch.cat([d, -d, norm], dim=-1)   # (bs, ne, 2*space+1)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden=128, n_hidden=1,
                 layer_norm=True, act='SiLU'):
        super().__init__()
        Act = {'SiLU': nn.SiLU, 'GELU': nn.GELU, 'ReLU': nn.ReLU}[act]
        layers = [nn.Linear(in_size, hidden), Act()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), Act()]
        layers.append(nn.Linear(hidden, out_size))
        if layer_norm:
            layers.append(nn.LayerNorm(out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Shared autoregressive mixin
# ===========================================================================

class _AutoregressiveMixin:
    """
    Drop-in autoregressive() that works for any subclass implementing forward().
    Identical rollout logic to physgto.py.
    """

    def autoregressive(self,
                       state_in,
                       node_pos,
                       edges,
                       time_seq,
                       conditions,
                       dt=None,
                       check_point=False):
        """
        Parameters
        ----------
        state_in  : (bs, N, in_dim)
        node_pos  : (bs, N, space_size)
        edges     : (bs, ne, 2)
        time_seq  : (bs, T, 1)   – raw time values per step
        conditions: (bs, cond_dim)
        dt        : scalar or (bs,) or None  – falls back to self.dt
        check_point: bool or int

        Returns
        -------
        outputs   : (bs, T, N, out_dim)
        """
        state_t = state_in
        outputs = []
        T = time_seq.shape[1]

        # Pre-compute static Fourier embeddings (shared across time steps)
        pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
        c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        for t in range(T):
            time_i = time_seq[:, t]   # (bs, 1)

            def _step(s_t, t_i, _pos=pos_enc, _c=c_enc):
                return self.forward(s_t, node_pos, edges, t_i,
                                    conditions, _pos, _c, dt)

            use_ckpt = (check_point is True or
                        (isinstance(check_point, int) and t >= check_point))

            if use_ckpt:
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                state_t = checkpoint(_step, state_t, time_i, use_reentrant=False)
            else:
                state_t = _step(state_t, time_i)

            outputs.append(state_t)

        return torch.stack(outputs, dim=1)   # (bs, T, N, out_dim)


# ===========================================================================
# Model 1 – GraphViT
# ===========================================================================
#
# Architecture (faithful to GraphViT / EAGLE baselines used in PhysGTO paper)
#   Encoder  →  n_gnn_layers × GNN blocks  →  n_attn_layers × Transformer
#   →  Decoder MLP  →  Euler step
#
# Works on ANY graph topology; no grid assumption.
# ===========================================================================

class _GNNBlock(nn.Module):
    """Single message-passing block (residual)."""

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.f_edge = MLP(node_dim * 2 + edge_dim, edge_dim,
                          hidden=node_dim, layer_norm=True)
        self.f_node = MLP(node_dim + edge_dim, node_dim,
                          hidden=node_dim, layer_norm=True)

    def forward(self, V, E, edges):
        bs, N, nd = V.shape
        ed = E.shape[-1]

        # valid_mask: (bs, ne) — False for -1-padded rows
        valid_mask = (edges >= 0).all(-1)  # (bs, ne)

        safe = edges.clamp(min=0, max=N - 1)

        # Gather sender / receiver features
        s_idx = safe[..., 0:1].expand(-1, -1, nd)
        r_idx = safe[..., 1:2].expand(-1, -1, nd)
        S = torch.gather(V, 1, s_idx)
        R = torch.gather(V, 1, r_idx)

        # Edge update — zero-out padded rows so they don't corrupt scatter
        e_new = self.f_edge(torch.cat([S, R, E], dim=-1))  # (bs,ne,ed)
        e_new = e_new * valid_mask.unsqueeze(-1)

        # Aggregate to receiver nodes (scatter mean over valid edges only)
        recv_idx = safe[..., 1:2].expand(-1, -1, ed)
        agg = scatter_mean(e_new * valid_mask.unsqueeze(-1), recv_idx, dim=1, dim_size=N)  # (bs,N,ed)

        # Node update (residual)
        v_new = self.f_node(torch.cat([V, agg], dim=-1))
        return V + v_new, E + e_new


class GraphViTModel(_AutoregressiveMixin, nn.Module):
    """
    Graph-Transformer baseline.

    Parameters
    ----------
    space_size    : spatial dimension (3 for LPBF)
    in_dim        : number of input state features per node
    out_dim       : number of output state features per node
    enc_dim       : internal channel width
    pos_enc_dim   : Fourier embedding frequency bands
    cond_dim      : dimension of the condition vector
    n_gnn_layers  : number of GNN message-passing layers
    n_attn_layers : number of Transformer encoder layers
    n_heads       : attention heads
    dt            : default time step for Euler integration
    """

    def __init__(self,
                 space_size=3,
                 in_dim=4,
                 out_dim=4,
                 enc_dim=128,
                 pos_enc_dim=5,
                 cond_dim=32,
                 n_gnn_layers=4,
                 n_attn_layers=4,
                 n_heads=4,
                 dt: float = 0.05):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.dt = dt

        # Derived embedding dims (same formulae as physgto.py)
        enc_s_dim = space_size + 2 * pos_enc_dim * space_size   # per-node pos emb
        enc_t_dim = 1 + 2 * pos_enc_dim                          # time emb
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim             # condition emb
        edge_raw  = 2 * space_size + 1                           # raw edge feature dim

        # ── Encoders ──────────────────────────────────────────────────────
        self.fv = MLP(in_dim + enc_s_dim, enc_dim, hidden=enc_dim, layer_norm=False)
        self.ft = MLP(enc_t_dim,          enc_dim, hidden=enc_dim, layer_norm=False)
        self.fc = MLP(enc_c_dim,          enc_dim, hidden=enc_dim, layer_norm=False)
        self.fe = MLP(edge_raw,           enc_dim, hidden=enc_dim, n_hidden=1,
                      layer_norm=False)

        # ── GNN layers ────────────────────────────────────────────────────
        self.gnn_layers = nn.ModuleList([
            _GNNBlock(enc_dim, enc_dim) for _ in range(n_gnn_layers)
        ])

        # ── Transformer encoder ───────────────────────────────────────────
        tf_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=n_heads,
            dim_feedforward=2 * enc_dim,
            batch_first=True,
            norm_first=True,          # Pre-LN (more stable)
            dropout=0.0,
        )
        # enable_flash_sdp: PyTorch ≥2.0 uses Flash Attention automatically
        # when batch_first=True and no custom attn_mask — no N×N matrix stored
        self.transformer = nn.TransformerEncoder(tf_layer,
                                                 num_layers=n_attn_layers,
                                                 enable_nested_tensor=False)

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
            nn.Linear(enc_dim, out_dim),
        )

    # ------------------------------------------------------------------
    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        """
        Parameters
        ----------
        state_in   : (bs, N, in_dim)
        node_pos   : (bs, N, space_size)
        edges      : (bs, ne, 2)  long or will be cast
        time_i     : (bs, 1)      raw time value
        conditions : (bs, cond_dim)
        pos_enc    : (bs, N, enc_s_dim)  pre-computed; computed here if None
        c_enc      : (bs, enc_c_dim)    pre-computed; computed here if None
        dt         : override time step

        Returns
        -------
        state_pred : (bs, N, out_dim)
        """
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
            c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim)   # (bs, enc_t_dim)
        edges_long = edges.long() if edges.dtype != torch.long else edges

        # Node embedding: state + positional encoding + time + condition
        node_feat = torch.cat([state_in, pos_enc], dim=-1)       # (bs,N,in+enc_s)
        V = self.fv(node_feat)                                    # (bs,N,enc_dim)
        V = V + self.ft(t_enc).unsqueeze(1)                       # broadcast time
        V = V + self.fc(c_enc).unsqueeze(1)                       # broadcast cond

        # Edge embedding
        E = self.fe(get_edge_info(edges_long, node_pos))          # (bs,ne,enc_dim)

        # GNN message passing
        for gnn in self.gnn_layers:
            V, E = gnn(V, E, edges_long)

        # Transformer over node tokens
        V = self.transformer(V)                                   # (bs,N,enc_dim)

        # Decode velocity field → Euler integration
        v_pred = self.decoder(V)                                  # (bs,N,out_dim)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)

        return state_in + dt * v_pred


# ===========================================================================
# Model 2 – FNO3D  (3-D Fourier Neural Operator)
# ===========================================================================
#
# Classic baseline from Li et al. 2021.
# Requires uniform cubic node ordering: nodes indexed in C-order over (D,H,W).
# LPBF数据天然满足此条件。
# ===========================================================================

class _SpectralConv3d(nn.Module):
    """Single 3-D spectral convolution layer."""

    def __init__(self, channels, modes1, modes2, modes3):
        super().__init__()
        scale = 1.0 / (channels ** 2)
        self.modes = (modes1, modes2, modes3)
        self.w = nn.Parameter(
            scale * torch.rand(channels, channels, modes1, modes2, modes3,
                               dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (bs, C, D, H, W)
        m1, m2, m3 = self.modes
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :m1, :m2, :m3] = torch.einsum(
            'bixyz,ioxyz->boxyz',
            x_ft[..., :m1, :m2, :m3], self.w
        )
        return torch.fft.irfftn(out_ft, s=x.shape[-3:])


class _FNOBlock3d(nn.Module):
    """FNO block = spectral conv + pointwise bypass + activation."""

    def __init__(self, channels, modes1, modes2, modes3):
        super().__init__()
        self.spectral = _SpectralConv3d(channels, modes1, modes2, modes3)
        self.bypass   = nn.Conv3d(channels, channels, 1)
        self.act      = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.bypass(x))


class FNO3DModel(_AutoregressiveMixin, nn.Module):
    """
    3-D Fourier Neural Operator baseline.

    Parameters
    ----------
    grid_shape  : (D, H, W) – must satisfy D*H*W == N
                  Node ordering in state_in must match C-order reshape.
    space_size  : 3 (fixed for 3-D LPBF)
    in_dim      : input state features per node
    out_dim     : output state features per node
    enc_dim     : FNO channel width (keep ≤ 64 for memory on large grids)
    pos_enc_dim : Fourier embedding frequency bands
    cond_dim    : condition vector dimension
    modes       : (m1, m2, m3) spectral truncation modes
    n_layers    : number of FNO blocks
    dt          : default Euler step

    Notes
    -----
    - Grid dimensions should be ≥ 2*max(modes) for spectral modes to fit.
    - For large grids (e.g., 128³), reduce enc_dim to 16–32.
    """

    def __init__(self,
                 grid_shape,
                 space_size=3,
                 in_dim=4,
                 out_dim=4,
                 enc_dim=32,
                 pos_enc_dim=5,
                 cond_dim=32,
                 modes=(8, 8, 8),
                 n_layers=4,
                 dt: float = 0.05):
        super().__init__()
        self.grid_shape  = tuple(grid_shape)
        self.pos_enc_dim = pos_enc_dim
        self.dt          = dt

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 1 + 2 * pos_enc_dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        node_in = in_dim + enc_s_dim    # per-node channels entering the grid

        # Global condition embedders → broadcast to all voxels
        self.t_embed = nn.Linear(enc_t_dim, enc_dim)
        self.c_embed = nn.Linear(enc_c_dim, enc_dim)

        # Lift input to enc_dim channels via 1×1×1 convolution
        self.lift = nn.Conv3d(node_in, enc_dim, 1)

        # FNO blocks
        m1, m2, m3 = modes
        self.blocks = nn.ModuleList([
            _FNOBlock3d(enc_dim, m1, m2, m3) for _ in range(n_layers)
        ])

        # Project to output
        self.project = nn.Sequential(
            nn.Conv3d(enc_dim, enc_dim, 1),
            nn.GELU(),
            nn.Conv3d(enc_dim, out_dim, 1),
        )

    # --- helpers ---
    def _to_grid(self, x):
        """(bs, N, C) → (bs, C, D, H, W)"""
        D, H, W = self.grid_shape
        return x.permute(0, 2, 1).reshape(x.shape[0], -1, D, H, W)

    def _from_grid(self, x):
        """(bs, C, D, H, W) → (bs, N, C)"""
        return x.flatten(2).permute(0, 2, 1)

    # ------------------------------------------------------------------
    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
            c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim)   # (bs, enc_t_dim)

        # Per-node features (spatial encoding fused into node channels)
        node_feat = torch.cat([state_in, pos_enc], dim=-1)       # (bs, N, node_in)

        # Global embeddings broadcast to all voxels
        t_emb = self.t_embed(t_enc).view(-1, self._enc_dim, 1, 1, 1)  # (bs,C,1,1,1)
        c_emb = self.c_embed(c_enc).view(-1, self._enc_dim, 1, 1, 1)

        # Reshape to 3-D grid
        x = self._to_grid(node_feat)   # (bs, node_in, D, H, W)
        x = self.lift(x)               # (bs, enc_dim, D, H, W)
        x = x + t_emb + c_emb         # add global conditioning

        # FNO blocks
        for blk in self.blocks:
            x = blk(x)

        # Project and Euler step
        v_pred = self._from_grid(self.project(x))   # (bs, N, out_dim)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)

        return state_in + dt * v_pred

    @property
    def _enc_dim(self):
        return self.lift.out_channels


# ===========================================================================
# Model 3 – UNet3DModel  (3-D U-Net)
# ===========================================================================
#
# Classic encoder-decoder CNN with skip connections.
# Best suited for LPBF because the active melt-pool region is small and
# spatially localised – U-Net's multi-scale skip connections preserve
# fine-grained local gradients while the bottleneck captures global context.
# ===========================================================================

class _DoubleConv3d(nn.Module):
    """Two consecutive 3×3×3 Conv + GroupNorm + SiLU blocks."""

    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        groups = min(groups, out_ch)          # safety for small channels
        # ensure out_ch divisible by groups
        while out_ch % groups != 0 and groups > 1:
            groups //= 2
        self.net = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class _Down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = _DoubleConv3d(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class _Up3d(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.conv = _DoubleConv3d(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch from non-power-of-2 grids
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear',
                              align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet3DModel(_AutoregressiveMixin, nn.Module):
    """
    3-D U-Net baseline.

    Parameters
    ----------
    grid_shape  : (D, H, W) – must satisfy D*H*W == N
    space_size  : 3 (fixed for 3-D LPBF)
    in_dim      : input state features per node
    out_dim     : output state features per node
    base_ch     : base channel width (doubles at each down-scale level)
    pos_enc_dim : Fourier embedding frequency bands
    cond_dim    : condition vector dimension
    n_levels    : number of down/up scale levels (depth of the U)
    dt          : default Euler step

    Notes
    -----
    - Grid dimensions should be divisible by 2^n_levels.
      If not, _Up3d handles the mismatch via interpolation.
    - base_ch=32, n_levels=3  →  channel widths: 32→64→128→256 (bottleneck)
    - Recommended: base_ch ≤ 32 for grids larger than 64³ (memory).
    """

    def __init__(self,
                 grid_shape,
                 space_size=3,
                 in_dim=4,
                 out_dim=4,
                 base_ch=32,
                 pos_enc_dim=5,
                 cond_dim=32,
                 n_levels=3,
                 dt: float = 0.05):
        super().__init__()
        self.grid_shape  = tuple(grid_shape)
        self.pos_enc_dim = pos_enc_dim
        self.dt          = dt

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 1 + 2 * pos_enc_dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        node_in = in_dim + enc_s_dim    # per-node channels

        # Channel widths at each level: [32, 64, 128, 256] for n_levels=3
        chs = [base_ch * (2 ** i) for i in range(n_levels + 1)]

        # Global condition projection (→ base_ch, added after first conv)
        self.t_embed = nn.Linear(enc_t_dim, base_ch)
        self.c_embed = nn.Linear(enc_c_dim, base_ch)

        # Encoder
        self.inc   = _DoubleConv3d(node_in, chs[0])
        self.downs = nn.ModuleList([
            _Down3d(chs[i], chs[i + 1]) for i in range(n_levels)
        ])

        # Decoder (reverse order)
        self.ups = nn.ModuleList([
            _Up3d(chs[i + 1], chs[i], chs[i])
            for i in range(n_levels - 1, -1, -1)
        ])

        # Output 1×1×1 conv
        self.outc = nn.Conv3d(chs[0], out_dim, 1)

    # --- helpers ---
    def _to_grid(self, x):
        D, H, W = self.grid_shape
        return x.permute(0, 2, 1).reshape(x.shape[0], -1, D, H, W)

    def _from_grid(self, x):
        return x.flatten(2).permute(0, 2, 1)

    # ------------------------------------------------------------------
    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos, 0, self.pos_enc_dim)
            c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim)   # (bs, enc_t_dim)

        # Global condition embeddings → (bs, base_ch, 1, 1, 1)
        t_emb = self.t_embed(t_enc).view(-1, self.t_embed.out_features, 1, 1, 1)
        c_emb = self.c_embed(c_enc).view(-1, self.c_embed.out_features, 1, 1, 1)

        # Per-node features → 3-D grid
        node_feat = torch.cat([state_in, pos_enc], dim=-1)
        x = self._to_grid(node_feat)   # (bs, node_in, D, H, W)

        # Encode initial features and inject global conditioning
        x = self.inc(x) + t_emb + c_emb   # (bs, base_ch, D, H, W)

        # ── Encoder path ──────────────────────────────────────────────
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # skips[-1] is the bottleneck; remove it before decoding
        skips.pop()

        # ── Decoder path ──────────────────────────────────────────────
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 1)])

        # Project and Euler step
        v_pred = self._from_grid(self.outc(x))    # (bs, N, out_dim)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)

        return state_in + dt * v_pred


# ===========================================================================
# Quick sanity check
# ===========================================================================

if __name__ == '__main__':
    import time

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ── Shared dummy data ──────────────────────────────────────────────────
    bs        = 2
    N         = 8 * 8 * 8          # 512 nodes = 8³ grid
    space     = 3
    in_dim    = 4
    out_dim   = 4
    cond_dim  = 16
    T         = 5
    D, H, W   = 8, 8, 8

    state_in   = torch.randn(bs, N, in_dim,  device=device)
    node_pos   = torch.rand( bs, N, space,   device=device)
    conditions = torch.rand( bs, cond_dim,   device=device)
    time_seq   = torch.rand( bs, T, 1,       device=device)

    # Build a minimal KNN edge set (each node → 6 nearest neighbours)
    from torch import randint
    ne = N * 6
    edges = torch.stack([
        torch.arange(N).repeat(6),
        randint(0, N, (ne,))
    ], dim=-1).unsqueeze(0).expand(bs, -1, -1).to(device)

    # ─────────────────────────────────────────────────────────────────────
    print("=" * 60)
    for Model, kwargs, name in [
        (GraphViTModel,
         dict(space_size=3, in_dim=in_dim, out_dim=out_dim,
              enc_dim=64, pos_enc_dim=4, cond_dim=cond_dim,
              n_gnn_layers=2, n_attn_layers=2, n_heads=4),
         "GraphViT"),
        (FNO3DModel,
         dict(grid_shape=(D, H, W), space_size=3,
              in_dim=in_dim, out_dim=out_dim,
              enc_dim=16, pos_enc_dim=4, cond_dim=cond_dim,
              modes=(4, 4, 4), n_layers=4),
         "FNO3D"),
        (UNet3DModel,
         dict(grid_shape=(D, H, W), space_size=3,
              in_dim=in_dim, out_dim=out_dim,
              base_ch=16, pos_enc_dim=4, cond_dim=cond_dim,
              n_levels=2),
         "UNet3D"),
    ]:
        model = Model(**kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        # Single forward
        t0 = time.time()
        with torch.no_grad():
            pred = model(state_in, node_pos, edges,
                         time_seq[:, 0], conditions)
        t1 = time.time()
        assert pred.shape == (bs, N, out_dim), f"Shape mismatch: {pred.shape}"

        # Autoregressive rollout
        with torch.no_grad():
            outputs = model.autoregressive(state_in, node_pos, edges,
                                           time_seq, conditions)
        assert outputs.shape == (bs, T, N, out_dim)

        print(f"[{name:12s}]  params={n_params:,}  "
              f"forward={1000*(t1-t0):.1f} ms  "
              f"output_shape={tuple(outputs.shape)}  ✓")

    print("=" * 60)
    print("All sanity checks passed.")
