"""
lpbf_baseline_models.py
=======================
Three LPBF-domain surrogate models, each with interface IDENTICAL to physgto.py:

    forward(state_in, node_pos, edges, time_i, conditions,
            pos_enc=None, c_enc=None, dt=None) -> state_pred

    autoregressive(state_in, node_pos, edges, time_seq, conditions,
                   dt=None, check_point=False) -> outputs  # (bs, T, N, out_dim)

Literature references
---------------------
Model 1 · MeltPoolResNet
    Hemmasian et al., "Surrogate Modeling of Melt Pool Temperature Field
    using Deep Learning," Additive Manufacturing Letters, 5:100123, 2023.
    GitHub: BaratiLab/SurrogateMeltPool_DL
    Architecture: 3-D residual CNN encoder–decoder that directly maps
    (current field, process params, time) → next temperature field on a
    uniform cubic voxel grid. The most-cited LPBF thermal field surrogate.

Model 2 · ConvLSTMModel
    Tian et al., "Physics-informed machine learning-based real-time long-
    horizon temperature fields prediction in metallic additive manufacturing,"
    Communications Engineering, 4:134, 2025.
    Safari et al., "Physics-Informed Surrogates for Temperature Prediction
    of Multi-Tracks in Laser Powder Bed Fusion," arXiv:2502.01820, 2025.
    Architecture: 3-D Convolutional GRU (ConvGRU) encoder–decoder.
    Extends ConvLSTM (Shi et al. 2015) to 3-D spatial domains; dominant
    architecture for spatiotemporal field prediction in LPBF literature.

Model 3 · ResNet3DModel
    Samber et al., "High-fidelity surrogate modelling for geometric deviation
    prediction in laser powder bed fusion using in-process monitoring data,"
    Virtual and Physical Prototyping, 20:2523550, 2025.
    Architecture: 3-D ResNet (volumetric residual network) used as the
    primary CNN baseline in recent LPBF monitoring/prediction benchmarks.

All three require grid-based node layout:
    grid_shape = (D, H, W) with D*H*W == N, nodes in C-order (Z-major).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ===========================================================================
# Shared utilities  (identical signatures to physgto.py)
# ===========================================================================

def FourierEmbedding(pos, pos_start, pos_length):
    """F(x) = [cos(2^i * pi * x), sin(2^i * pi * x), x]  (same as physgto.py)"""
    original_shape = pos.shape
    p = pos.reshape(-1, original_shape[-1])
    idx  = torch.arange(pos_start, pos_start + pos_length, device=pos.device).float()
    freq = 2 ** idx * torch.pi
    cos_f = torch.cos(freq[None, None, :] * p.unsqueeze(-1))
    sin_f = torch.sin(freq[None, None, :] * p.unsqueeze(-1))
    emb = torch.cat([cos_f, sin_f], dim=-1).reshape(*original_shape[:-1], -1)
    return torch.cat([emb, pos], dim=-1)


def _make_gn(channels, groups=8):
    """GroupNorm with automatic group-count fallback for small channel counts."""
    g = groups
    while channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, channels)


# ===========================================================================
# Shared autoregressive mixin  (same rollout logic as physgto.py)
# ===========================================================================

class _AutoregressiveMixin:
    """
    Drop-in autoregressive() for any subclass that implements forward().
    Logic is byte-for-byte equivalent to physgto.py's autoregressive().
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
        state_in   : (bs, N, in_dim)
        node_pos   : (bs, N, space_size)
        edges      : (bs, ne, 2)
        time_seq   : (bs, T, 1)   raw time at each step
        conditions : (bs, cond_dim)
        dt         : scalar / (bs,) / None
        check_point: bool or int

        Returns
        -------
        (bs, T, N, out_dim)
        """
        state_t = state_in
        outputs  = []
        T = time_seq.shape[1]

        pos_enc = FourierEmbedding(node_pos,   0, self.pos_enc_dim)
        c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        for t in range(T):
            time_i = time_seq[:, t]

            def _step(s, ti, _p=pos_enc, _c=c_enc):
                return self.forward(s, node_pos, edges, ti, conditions, _p, _c, dt)

            use_ckpt = (check_point is True or
                        (isinstance(check_point, int) and t >= check_point))
            if use_ckpt:
                if not state_t.requires_grad and state_t.is_floating_point():
                    state_t.requires_grad_()
                state_t = checkpoint(_step, state_t, time_i, use_reentrant=False)
            else:
                state_t = _step(state_t, time_i)

            outputs.append(state_t)

        return torch.stack(outputs, dim=1)


# ===========================================================================
# Helpers shared by all three models
# ===========================================================================

class _ConditionInjector(nn.Module):
    """
    Project (time, condition) embeddings → single channel vector (bs, ch)
    for adding to voxel feature maps via broadcasting (bs, ch, 1, 1, 1).
    """
    def __init__(self, enc_t_dim, enc_c_dim, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(enc_t_dim + enc_c_dim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, t_enc, c_enc):
        # t_enc: (bs, enc_t_dim), c_enc: (bs, enc_c_dim)
        return self.proj(torch.cat([t_enc, c_enc], dim=-1))    # (bs, out_ch)


# ===========================================================================
# Model 1 – MeltPoolResNet
# ===========================================================================
#
# Reference: Hemmasian et al., Additive Manufacturing Letters, 2023.
# Code:      BaratiLab/SurrogateMeltPool_DL (GitHub)
#
# Original design: a 3D CNN that takes process parameters and time step as
# input (not a field) and outputs the full 3D temperature field snapshot.
# Adapted here for our autoregressive setting: the current state field is
# concatenated with per-voxel positional encoding and the global time /
# condition embeddings are injected via FiLM (feature-wise linear modulation),
# then residual 3D-conv blocks transform it to the velocity field Δu/Δt.
# ===========================================================================

class _ResBlock3d(nn.Module):
    """3-D residual block with optional FiLM conditioning."""

    def __init__(self, ch, cond_ch=None, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv3d(ch, ch, kernel, padding=pad, bias=False)
        self.gn1   = _make_gn(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel, padding=pad, bias=False)
        self.gn2   = _make_gn(ch)
        self.act   = nn.SiLU()

        # FiLM: predict scale & shift from condition
        self.film = None
        if cond_ch is not None:
            self.film = nn.Linear(cond_ch, 2 * ch)

    def forward(self, x, cond=None):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        if self.film is not None and cond is not None:
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = h * (1 + gamma.view(-1, h.shape[1], 1, 1, 1)) \
                  + beta.view(-1, h.shape[1], 1, 1, 1)
        return self.act(x + h)


class MeltPoolResNet(_AutoregressiveMixin, nn.Module):
    """
    3-D Residual CNN surrogate  (Hemmasian et al. 2023, adapted for autoregressive use).

    Parameters
    ----------
    grid_shape  : (D, H, W) – must satisfy D*H*W == N
    space_size  : spatial dimension (3 for LPBF)
    in_dim      : number of input physical fields per node
    out_dim     : number of output physical fields per node
    base_ch     : base channel count (doubled twice in the encoder)
    pos_enc_dim : Fourier frequency bands for positional/time/condition embedding
    cond_dim    : dimension of the process condition vector
    n_res       : number of residual blocks per encoder/decoder level
    dt          : default Euler integration step
    """

    def __init__(self,
                 grid_shape,
                 space_size=3,
                 in_dim=4,
                 out_dim=4,
                 base_ch=32,
                 pos_enc_dim=5,
                 cond_dim=32,
                 n_res=2,
                 dt: float = 0.05):
        super().__init__()
        self.grid_shape  = tuple(grid_shape)
        self.pos_enc_dim = pos_enc_dim
        self.dt          = dt

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size   # per-node pos emb
        enc_t_dim = 1 + 2 * pos_enc_dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        node_in  = in_dim + enc_s_dim       # channels at voxel level
        cond_ch  = base_ch                  # FiLM condition dimension
        ch0, ch1, ch2 = base_ch, base_ch*2, base_ch*4

        # Global condition projector → single cond vector per batch
        self.cond_proj = _ConditionInjector(enc_t_dim, enc_c_dim, cond_ch)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(node_in, ch0, 3, padding=1, bias=False),
            _make_gn(ch0), nn.SiLU()
        )
        self.enc0 = nn.ModuleList([_ResBlock3d(ch0, cond_ch) for _ in range(n_res)])
        self.down0 = nn.Conv3d(ch0, ch1, 3, stride=2, padding=1, bias=False)

        self.enc1 = nn.ModuleList([_ResBlock3d(ch1, cond_ch) for _ in range(n_res)])
        self.down1 = nn.Conv3d(ch1, ch2, 3, stride=2, padding=1, bias=False)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([_ResBlock3d(ch2, cond_ch) for _ in range(n_res)])

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up1   = nn.ConvTranspose3d(ch2, ch1, 2, stride=2)
        self.dec1  = nn.ModuleList([_ResBlock3d(ch1*2, cond_ch) for _ in range(n_res)])
        self.proj1 = nn.Conv3d(ch1*2, ch1, 1)   # channel reduction after skip cat

        self.up0   = nn.ConvTranspose3d(ch1, ch0, 2, stride=2)
        self.dec0  = nn.ModuleList([_ResBlock3d(ch0*2, cond_ch) for _ in range(n_res)])
        self.proj0 = nn.Conv3d(ch0*2, ch0, 1)

        # Output 1×1×1 convolution
        self.outc  = nn.Conv3d(ch0, out_dim, 1)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _to_grid(self, x):
        """(bs, N, C) → (bs, C, D, H, W)"""
        D, H, W = self.grid_shape
        return x.permute(0, 2, 1).reshape(x.shape[0], -1, D, H, W)

    def _from_grid(self, x):
        """(bs, C, D, H, W) → (bs, N, C)"""
        return x.flatten(2).permute(0, 2, 1)

    def _run_blocks(self, x, blocks, cond):
        for b in blocks:
            x = b(x, cond)
        return x

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        """
        Inputs / outputs identical to physgto.py Model.forward().
        `edges` is accepted but ignored (grid model uses convolutions).
        """
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos,   0, self.pos_enc_dim)
            c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim)    # (bs, enc_t_dim)
        cond  = self.cond_proj(t_enc, c_enc)                      # (bs, base_ch)

        # Per-node features → 3-D grid
        node_feat = torch.cat([state_in, pos_enc], dim=-1)
        x = self._to_grid(node_feat)              # (bs, node_in, D, H, W)

        # ── Encoder ──
        x  = self.stem(x)                         # (bs, ch0, D, H, W)
        x0 = self._run_blocks(x, self.enc0, cond) # skip connection
        x  = self.down0(x0)                       # (bs, ch1, D/2, H/2, W/2)

        x1 = self._run_blocks(x, self.enc1, cond) # skip connection
        x  = self.down1(x1)                       # (bs, ch2, D/4, H/4, W/4)

        # ── Bottleneck ──
        x  = self._run_blocks(x, self.bottleneck, cond)

        # ── Decoder ──
        x  = self.up1(x)                          # (bs, ch1, D/2, H/2, W/2)
        if x.shape != x1.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x  = self._run_blocks(torch.cat([x, x1], dim=1), self.dec1, cond)
        x  = self.proj1(x)

        x  = self.up0(x)                          # (bs, ch0, D, H, W)
        if x.shape != x0.shape:
            x = F.interpolate(x, size=x0.shape[2:], mode='trilinear', align_corners=False)
        x  = self._run_blocks(torch.cat([x, x0], dim=1), self.dec0, cond)
        x  = self.proj0(x)

        # ── Output + Euler step ──
        v_pred = self._from_grid(self.outc(x))    # (bs, N, out_dim)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)

        return state_in + dt * v_pred


# ===========================================================================
# Model 2 – ConvLSTMModel  (3-D ConvGRU)
# ===========================================================================
#
# References:
#   Tian et al., Communications Engineering, 4:134, 2025.
#   Safari et al., arXiv 2502.01820, 2025.
#   Samber et al., Virtual and Physical Prototyping, 2025.
#   (Original ConvLSTM: Shi et al., NeurIPS, 2015.)
#
# The ConvLSTM/ConvGRU family is the dominant temporal CNN baseline in LPBF
# surrogate literature. We implement a 3-D ConvGRU cell (lighter than LSTM)
# in an encoder–decoder topology matching the convention in the LPBF papers.
# For the physgto autoregressive interface, each forward() call runs K
# internal ConvGRU steps initialised from the current state; hidden state
# is NOT persisted across calls (consistent with the stateless physgto API).
# ===========================================================================

class _ConvGRUCell3D(nn.Module):
    """
    3-D Convolutional GRU cell.
    Gates use 3×3×3 convolutions over (D, H, W) spatial dimensions.
    """

    def __init__(self, in_ch, h_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        # Reset gate, update gate (jointly computed)
        self.conv_rz = nn.Conv3d(in_ch + h_ch, 2 * h_ch, kernel, padding=pad, bias=True)
        # Candidate hidden state
        self.conv_n  = nn.Conv3d(in_ch + h_ch, h_ch,     kernel, padding=pad, bias=True)
        self.h_ch    = h_ch

    def forward(self, x, h):
        """
        x : (bs, in_ch, D, H, W)
        h : (bs, h_ch,  D, H, W)  – previous hidden state (zero on first call)
        Returns updated h of same shape.
        """
        rz = torch.sigmoid(self.conv_rz(torch.cat([x, h], dim=1)))
        r, z = rz.chunk(2, dim=1)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * n
        return h_new


class ConvLSTMModel(_AutoregressiveMixin, nn.Module):
    """
    3-D ConvGRU encoder–decoder surrogate for LPBF thermal field prediction.

    Parameters
    ----------
    grid_shape   : (D, H, W) — must satisfy D*H*W == N
    space_size   : 3 for LPBF
    in_dim       : input physical channels per node
    out_dim      : output physical channels per node
    base_ch      : base channel count
    pos_enc_dim  : Fourier frequency bands
    cond_dim     : process condition dimension
    n_gru_steps  : number of internal GRU recurrence steps per call
                   (k=1 → pure gated CNN; k>1 → iterative temporal refinement)
    dt           : default Euler step
    """

    def __init__(self,
                 grid_shape,
                 space_size=3,
                 in_dim=4,
                 out_dim=4,
                 base_ch=32,
                 pos_enc_dim=5,
                 cond_dim=32,
                 n_gru_steps=2,
                 dt: float = 0.05):
        super().__init__()
        self.grid_shape   = tuple(grid_shape)
        self.pos_enc_dim  = pos_enc_dim
        self.n_gru_steps  = n_gru_steps
        self.dt           = dt

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 1 + 2 * pos_enc_dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        node_in = in_dim + enc_s_dim
        ch0, ch1 = base_ch, base_ch * 2
        cond_out = base_ch

        # Global condition injector (time + process params)
        self.cond_proj = _ConditionInjector(enc_t_dim, enc_c_dim, cond_out)

        # ── Encoder (lifting + downscale) ──────────────────────────────────
        self.lift = nn.Sequential(
            nn.Conv3d(node_in + 1, ch0, 3, padding=1, bias=False),   # +1 for cond broadcast
            _make_gn(ch0), nn.SiLU()
        )
        # Downscale to ch1 → bottleneck GRU hidden size
        self.enc_down = nn.Conv3d(ch0, ch1, 3, stride=2, padding=1, bias=False)

        # ── ConvGRU cells at two scales ────────────────────────────────────
        self.gru0 = _ConvGRUCell3D(ch0, ch0)    # full resolution
        self.gru1 = _ConvGRUCell3D(ch1, ch1)    # half resolution (bottleneck)

        # ── Decoder ────────────────────────────────────────────────────────
        self.dec_up = nn.ConvTranspose3d(ch1, ch0, 2, stride=2)
        self.dec_conv = nn.Sequential(
            nn.Conv3d(ch0 * 2, ch0, 3, padding=1, bias=False),
            _make_gn(ch0), nn.SiLU()
        )
        self.outc = nn.Conv3d(ch0, out_dim, 1)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _to_grid(self, x):
        D, H, W = self.grid_shape
        return x.permute(0, 2, 1).reshape(x.shape[0], -1, D, H, W)

    def _from_grid(self, x):
        return x.flatten(2).permute(0, 2, 1)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos,   0, self.pos_enc_dim)
            c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim)
        cond  = self.cond_proj(t_enc, c_enc)                   # (bs, cond_out)

        # Per-node features → 3-D grid
        node_feat = torch.cat([state_in, pos_enc], dim=-1)
        xg = self._to_grid(node_feat)                          # (bs, node_in, D, H, W)

        # Broadcast cond as an extra spatial channel
        cond_scalar = cond.mean(dim=1)                                    # (bs,)
        cond_map = cond_scalar[:, None, None, None, None].expand(-1, 1, *xg.shape[2:])
        xg = torch.cat([xg, cond_map], dim=1)                  # (bs, node_in+1, D,H,W)

        # ── Encoder ──
        x0 = self.lift(xg)                                     # (bs, ch0, D, H, W)
        x1 = self.enc_down(x0)                                 # (bs, ch1, D/2,H/2,W/2)

        # ── ConvGRU recurrence (initialised from encoded input, zero h0) ──
        h0 = torch.zeros_like(x0)
        h1 = torch.zeros_like(x1)
        for _ in range(self.n_gru_steps):
            h0 = self.gru0(x0, h0)
            h1 = self.gru1(x1, h1)

        # ── Decoder ──
        h1_up = self.dec_up(h1)                                # (bs, ch0, D, H, W)
        if h1_up.shape != h0.shape:
            h1_up = F.interpolate(h1_up, size=h0.shape[2:],
                                  mode='trilinear', align_corners=False)
        x = self.dec_conv(torch.cat([h1_up, h0], dim=1))      # (bs, ch0, D, H, W)

        # ── Output + Euler step ──
        v_pred = self._from_grid(self.outc(x))                 # (bs, N, out_dim)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)

        return state_in + dt * v_pred


# ===========================================================================
# Model 3 – ResNet3D
# ===========================================================================
#
# Reference: Samber et al., "High-fidelity surrogate modelling for geometric
# deviation prediction in laser powder bed fusion using in-process monitoring
# data," Virtual and Physical Prototyping, 20:2523550, 2025.
#
# The 3-D ResNet processes entire volumetric inputs, making it well-suited
# for LPBF's structured 3-D voxel grid. We implement a lightweight version
# that matches the description in the paper (volumetric residual blocks,
# no down/upsampling — operates at full resolution via large receptive fields
# built from stacked 3×3×3 convolutions).
# ===========================================================================

class _Res3DBlock(nn.Module):
    """
    3-D pre-activation residual block (He et al. 2016 v2 style).
    Uses dilated 3×3×3 convolutions to increase receptive field efficiently.
    """

    def __init__(self, ch, dilation=1):
        super().__init__()
        pad = dilation
        self.gn1   = _make_gn(ch)
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.gn2   = _make_gn(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.act   = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.gn1(x)))
        h = self.conv2(self.act(self.gn2(h)))
        return x + h


class ResNet3DModel(_AutoregressiveMixin, nn.Module):
    """
    3-D Residual Network surrogate for LPBF (Samber et al. 2025).
    Operates at full voxel grid resolution using dilated convolutions to
    capture the large thermal diffusion length-scales without downsampling.

    Parameters
    ----------
    grid_shape  : (D, H, W) — must satisfy D*H*W == N
    space_size  : 3 for LPBF
    in_dim      : input physical channels per node
    out_dim     : output physical channels per node
    ch          : channel count throughout (uniform width)
    pos_enc_dim : Fourier frequency bands
    cond_dim    : process condition dimension
    n_blocks    : total number of residual blocks
    dilations   : dilation schedule (cycles if len < n_blocks)
                  e.g., [1, 2, 4] grows receptive field with fewer params
    dt          : default Euler step
    """

    def __init__(self,
                 grid_shape,
                 space_size=3,
                 in_dim=4,
                 out_dim=4,
                 ch=64,
                 pos_enc_dim=5,
                 cond_dim=32,
                 n_blocks=8,
                 dilations=(1, 2, 4),
                 dt: float = 0.05):
        super().__init__()
        self.grid_shape  = tuple(grid_shape)
        self.pos_enc_dim = pos_enc_dim
        self.dt          = dt

        enc_s_dim = space_size + 2 * pos_enc_dim * space_size
        enc_t_dim = 1 + 2 * pos_enc_dim
        enc_c_dim = (1 + 2 * pos_enc_dim) * cond_dim

        node_in  = in_dim + enc_s_dim

        # Global condition projector
        self.cond_proj = _ConditionInjector(enc_t_dim, enc_c_dim, ch)

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(node_in, ch, 3, padding=1, bias=False),
            _make_gn(ch), nn.SiLU()
        )

        # ── Residual blocks ───────────────────────────────────────────────────
        dil_cycle = list(dilations)
        self.blocks = nn.ModuleList([
            _Res3DBlock(ch, dilation=dil_cycle[i % len(dil_cycle)])
            for i in range(n_blocks)
        ])

        # ── Head ─────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            _make_gn(ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 1),
            nn.SiLU(),
            nn.Conv3d(ch, out_dim, 1),
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _to_grid(self, x):
        D, H, W = self.grid_shape
        return x.permute(0, 2, 1).reshape(x.shape[0], -1, D, H, W)

    def _from_grid(self, x):
        return x.flatten(2).permute(0, 2, 1)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, state_in, node_pos, edges, time_i, conditions,
                pos_enc=None, c_enc=None, dt=None):
        if pos_enc is None or c_enc is None:
            pos_enc = FourierEmbedding(node_pos,   0, self.pos_enc_dim)
            c_enc   = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        t_enc = FourierEmbedding(time_i, 0, self.pos_enc_dim)
        cond  = self.cond_proj(t_enc, c_enc)                   # (bs, ch)

        # Per-node features → 3-D grid
        node_feat = torch.cat([state_in, pos_enc], dim=-1)
        x = self._to_grid(node_feat)                           # (bs, node_in, D,H,W)

        # ── Stem + cond injection ──
        x = self.stem(x)                                        # (bs, ch, D, H, W)
        cond_map = cond.view(-1, cond.shape[-1], 1, 1, 1)
        x = x + cond_map                                        # FiLM-lite: additive

        # ── Residual blocks ──
        for blk in self.blocks:
            x = blk(x)

        # ── Output + Euler step ──
        v_pred = self._from_grid(self.head(x))                 # (bs, N, out_dim)

        if dt is None:
            dt = self.dt
        elif isinstance(dt, torch.Tensor) and dt.dim() == 1:
            dt = dt.view(-1, 1, 1)

        return state_in + dt * v_pred


# ===========================================================================
# Dimension validation + sanity check
# ===========================================================================

if __name__ == '__main__':
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    bs, D, H, W = 2, 8, 8, 8
    N = D * H * W        # 512 nodes
    space, in_dim, out_dim, cond_dim = 3, 4, 4, 16
    T = 5
    grid = (D, H, W)

    state_in   = torch.randn(bs, N, in_dim,  device=device)
    node_pos   = torch.rand( bs, N, space,   device=device)
    conditions = torch.rand( bs, cond_dim,   device=device)
    time_seq   = torch.rand( bs, T, 1,       device=device)
    edges      = torch.zeros(bs, 1, 2, dtype=torch.long, device=device)  # dummy

    configs = [
        (MeltPoolResNet,
         dict(grid_shape=grid, space_size=space, in_dim=in_dim, out_dim=out_dim,
              base_ch=16, pos_enc_dim=4, cond_dim=cond_dim, n_res=1),
         "MeltPoolResNet"),
        (ConvLSTMModel,
         dict(grid_shape=grid, space_size=space, in_dim=in_dim, out_dim=out_dim,
              base_ch=16, pos_enc_dim=4, cond_dim=cond_dim, n_gru_steps=2),
         "ConvLSTMModel"),
        (ResNet3DModel,
         dict(grid_shape=grid, space_size=space, in_dim=in_dim, out_dim=out_dim,
              ch=32, pos_enc_dim=4, cond_dim=cond_dim,
              n_blocks=4, dilations=(1, 2, 4)),
         "ResNet3D"),
    ]

    print("=" * 65)
    for Model, kwargs, name in configs:
        model = Model(**kwargs).to(device)
        nparams = sum(p.numel() for p in model.parameters())

        t0 = time.time()
        with torch.no_grad():
            pred = model(state_in, node_pos, edges, time_seq[:, 0], conditions)
        dt_fwd = time.time() - t0
        assert pred.shape == (bs, N, out_dim), f"forward shape mismatch: {pred.shape}"

        with torch.no_grad():
            out = model.autoregressive(state_in, node_pos, edges,
                                       time_seq, conditions)
        assert out.shape == (bs, T, N, out_dim)

        print(f"[{name:18s}]  params={nparams:>8,}  "
              f"fwd={1000*dt_fwd:>6.1f}ms  "
              f"rollout={tuple(out.shape)}  ✓")

    print("=" * 65)
    print("All LPBF baseline models passed sanity check.")
