"""
dataset_lpbf_v3.py — Slot-based Dataset for LPBF-NO v3

Implements the fixed-slot design from LPBF_NO_v3_design.md §2:
  - Predefined K field slots with presence_mask and field_type_embedding indices
  - Input dimension is fixed regardless of which fields are active in a given run
  - Outputs slot_values [T, N, K], presence_mask [K], active_indices [n_active]
  - Fully backward-compatible: existing `state` field still returned for legacy paths

Key differences from dataset_fast.py:
  - Accepts model.field_slots config (list of {name, type, default})
  - __getitem__ fills slot_values from actual field data; missing slots use default (0.0)
  - presence_mask marks which slots are available for a given run
  - Physical node_pos (un-normalized absolute coords) also returned for laser field
  - Laser parameters extracted from conditions for analytic field computation (if needed)

Coordinate convention (from LPBF_NO_v3_design.md §0):
  x = laser scan direction
  y = depth direction (air above, metal below)
  z = lateral (scan width) direction

Grid storage order in HDF5 (from dataset_fast.py:66-70):
  for z: for y: for x:  →  index = z * Nx * Ny + y * Nx + x
  → rearrange: 'b (hz hy hx) c -> b c hx hy hz'
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import ChannelNormalizer, build_active_mask


# ---------------------------------------------------------------------------
# Re-use utility functions from dataset_fast.py (import directly)
# ---------------------------------------------------------------------------

from src.dataset.dataset_fast import (
    _read_file_list,
    _normalize_stride,
    _compute_downsample_indices,
    _build_grid_edges,
    _build_node_type,
    _process_condition_normalize,
    _condition_vector,
    _compute_stats,
    AeroGtoDataset,  # base class for meta-building helpers
)


# ---------------------------------------------------------------------------
# Slot metadata helpers
# ---------------------------------------------------------------------------

# Default field-type registry
FIELD_TYPE_REGISTRY: Dict[str, str] = {
    "T":              "temperature",
    "Ux":             "velocity",
    "Uy":             "velocity",
    "Uz":             "velocity",
    "alpha.air":      "interface",
    "alpha.titanium": "interface",
    "gamma_liquid":   "interface",
}

# Default normalizer statistics (mirrors dataset_fast._load_normalizer)
FIELD_STATS_DEFAULT: Dict[str, Tuple[float, float]] = {
    "T":              (5.2999e+02, 4.5454e+02),
    "Ux":             (4.0041e-05, 2.4173e-01),
    "Uy":             (-1.6900e-05, 2.5172e-01),
    "Uz":             (3.3602e-07, 1.1976e-01),
    "alpha.air":      (0.0, 1.0),
    "alpha.titanium": (0.0, 1.0),
    "gamma_liquid":   (0.0, 1.0),
}


def _build_default_slots(fields: List[str]) -> List[Dict]:
    """Build a default slot config from a field list (type inferred from registry)."""
    slots = []
    for fname in fields:
        ftype = FIELD_TYPE_REGISTRY.get(fname, "generic")
        slots.append({"name": fname, "type": ftype, "default": 0.0})
    return slots


# ---------------------------------------------------------------------------
# LPBFSlotDataset
# ---------------------------------------------------------------------------

class LPBFSlotDataset(Dataset):
    """
    Slot-based LPBF dataset for LPBF-NO v3.

    Supports two operating modes based on config:
      1. `data.field_slots` defined: uses fixed-slot design (recommended)
      2. `data.field_slots` not defined: auto-generates slots from `data.fields`
         (identical to dataset_fast in terms of active fields)

    Each __getitem__ returns:
      state:            [T+1, N, K]   slot-aligned state (normalized)
      slot_values:      [T+1, N, K]   alias to state (for model clarity)
      presence_mask:    [K]           1.0 for active slots, 0.0 for inactive
      active_indices:   [n_active]    indices of active slots
      slot_types:       [K] (int)     0=temperature,1=interface,2=velocity,3=generic
      node_pos:         [N, 3]        normalized node positions
      node_pos_physical:[N, 3]        un-normalized physical coords (for laser field)
      edges:            [ne, 2]       grid edges
      node_type:        [N, 1]        boundary type
      spatial_inform:   [10]          grid/time metadata
      conditions:       [cond_dim]    normalized process parameters
      time_seq:         [T, 1]        relative time steps
      dt:               float         time step size
      grid_shape:       [3]           (Nx, Ny, Nz)

    Args:
        args:           config namespace (must have .data and optionally .model)
        mode:           "train" | "test"
        mat_data:       optional material normalization stats
        spatial_stride: override for test-time spatial stride
    """

    # Slot type → integer index for embedding
    SLOT_TYPE_TO_IDX: Dict[str, int] = {
        "temperature": 0,
        "interface":   1,
        "velocity":    2,
        "generic":     3,
    }

    def __init__(
        self,
        args,
        mode: str = "train",
        mat_data=None,
        spatial_stride=None,
    ):
        super().__init__()
        assert mode in {"train", "test"}
        data_cfg = args.data
        self.config = data_cfg
        self.mode = mode

        # ---- Fields actually predicted in this run ----
        self.fields = data_cfg.get("fields", ["T"])

        # ---- Slot configuration (fixed or auto) ----
        model_cfg = args.model if hasattr(args, "model") else {}
        raw_slots = model_cfg.get("field_slots", None)
        if raw_slots is None:
            # Auto-generate from fields (backward-compatible)
            raw_slots = _build_default_slots(self.fields)

        self.slot_names: List[str] = [s["name"] for s in raw_slots]
        self.slot_types_str: List[str] = [s.get("type", FIELD_TYPE_REGISTRY.get(s["name"], "generic")) for s in raw_slots]
        self.slot_defaults: List[float] = [float(s.get("default", 0.0)) for s in raw_slots]
        self.K = len(self.slot_names)

        # Integer type indices for nn.Embedding
        self.slot_type_indices: List[int] = [
            self.SLOT_TYPE_TO_IDX.get(t, 3) for t in self.slot_types_str
        ]

        # Which slots are active for this run (field name → slot index)
        self.field_to_slot: Dict[str, int] = {
            name: i for i, name in enumerate(self.slot_names)
            if name in self.fields
        }
        self.active_slot_indices: List[int] = [
            i for i, name in enumerate(self.slot_names)
            if name in self.fields
        ]

        # ---- Horizon / stride settings ----
        self.input_steps = data_cfg.get("input_steps", 1)
        self.horizon = data_cfg.get(f"horizon_{mode}", 1)
        self.pf_extra = data_cfg.get("horizon_pf_extra", 0) if mode == "train" else 0
        self.time_stride = data_cfg.get("time_stride", 1)

        if mode == "test" and spatial_stride is not None:
            self.spatial_stride = _normalize_stride(spatial_stride)
        else:
            self.spatial_stride = _normalize_stride(data_cfg.get("spatial_stride", 1))

        self.normalize = data_cfg.get("normalize", True)
        self.mat_mean_and_std = mat_data
        self.samples_per_file = data_cfg.get("samples_per_file", 32)
        self.norm_cache = data_cfg.get("norm_cache")
        self.time_ref = 2e-5 if data_cfg.get("dt_scale", False) else 1
        self.edge_sample_ratio = data_cfg.get("edge_sample_ratio", 1.0)

        self.mask_cfg = args.train.get(
            "weight_loss",
            {"field": ["T", "alpha.air"], "threshold": [800, [0.4, 0.6]]}
        )

        # ---- Build file meta cache ----
        self.file_paths = _read_file_list(data_cfg[f"{mode}_list"])
        self.meta_cache = {}
        self.sample_keys = []
        self.max_start_per_file = []

        for file_id, path in enumerate(self.file_paths):
            meta = self._build_meta(path)
            self.meta_cache[path] = meta
            self.max_start_per_file.append(meta["max_start"])

            if mode == "train":
                for _ in range(self.samples_per_file):
                    self.sample_keys.append((file_id, None))
            else:
                step = max(1, self.horizon // 2)
                for start in range(1, meta["max_start"] + 1, step):
                    self.sample_keys.append((file_id, start))

        example_meta = next(iter(self.meta_cache.values()))
        self.cond_dim = example_meta["conditions"].shape[-1]
        self.node_num = example_meta["node_pos"].shape[0]
        self.dt = example_meta["dt"] / self.time_ref

        # ---- Normalizer for the active fields ----
        if self.normalize and mode == "train":
            self.normalizer = self._load_normalizer()
        else:
            self.normalizer = ChannelNormalizer(
                np.zeros(len(self.fields), dtype=np.float32),
                np.ones(len(self.fields), dtype=np.float32),
            )

        # Sync and pre-cache scaled positions
        self._sync_norm_cache()

    # ------------------------------------------------------------------
    # Normalizer helpers
    # ------------------------------------------------------------------

    def _load_normalizer(self) -> ChannelNormalizer:
        mean_list, std_list = [], []
        for fname in self.fields:
            m, s = FIELD_STATS_DEFAULT.get(fname, (0.0, 1.0))
            mean_list.append(m)
            std_list.append(s)
        return ChannelNormalizer(
            np.array(mean_list, dtype=np.float32),
            np.array(std_list, dtype=np.float32),
        )

    def _sync_norm_cache(self):
        self.norm_mean = self.normalizer.mean   # [1, 1, n_active_fields]
        self.norm_std  = self.normalizer.std + self.normalizer.eps
        if self.normalize:
            for path, meta in self.meta_cache.items():
                meta["node_pos_scaled"] = self._scale_3D_pos(meta["node_pos"])

    @staticmethod
    def _scale_3D_pos(node_pos: torch.Tensor) -> torch.Tensor:
        pos_min = node_pos.min(dim=0).values
        pos_max = node_pos.max(dim=0).values
        return (node_pos - pos_min) / (pos_max - pos_min + 1e-8)

    # ------------------------------------------------------------------
    # Meta building (mirrors dataset_fast._build_meta)
    # ------------------------------------------------------------------

    def _build_meta(self, path: str) -> dict:
        path = str(Path(path).expanduser().resolve())
        with h5py.File(path, "r") as f:
            block = f["mesh/block"][0].astype(int)
            bound = f["mesh/bounds"][:].astype(np.float32)

            grid_shape = (block[0] + 1, block[1] + 1, block[2] + 1)
            indices, ds_shape = _compute_downsample_indices(grid_shape, self.spatial_stride)

            spatial_inform = torch.from_numpy(
                np.concatenate([
                    bound.flatten(),
                    np.array(ds_shape, dtype=np.float32),
                    np.array([self.time_ref], dtype=np.float32),
                ])
            )  # [10]

            point_all = f["point"][:]
            point = point_all[indices]
            node_pos = torch.from_numpy(point.astype(np.float32))  # physical coords

            edges    = _build_grid_edges(ds_shape, self.edge_sample_ratio)
            node_type = _build_node_type(ds_shape)

            if self.normalize:
                if self.mat_mean_and_std is None:
                    conditions, self.mat_mean_and_std = _process_condition_normalize(f)
                else:
                    conditions, _ = _process_condition_normalize(f, self.mat_mean_and_std)
            else:
                conditions = _condition_vector(f, self.fields)

            conditions = torch.from_numpy(conditions)

            time_all = f["time"][:]
            dt = np.float32(np.mean(np.diff(time_all)))
            total_steps = len(time_all)
            max_start = total_steps - (
                self.input_steps + (self.horizon + self.pf_extra) * self.time_stride
            )
            if max_start < 0:
                raise ValueError(
                    f"Time window exceeds data length: total_steps={total_steps}, "
                    f"required={self.input_steps + (self.horizon + self.pf_extra) * self.time_stride}"
                )

        return {
            "grid_shape":  grid_shape,
            "indices":     indices,
            "ds_shape":    ds_shape,
            "node_pos":    node_pos,      # physical (un-normalized)
            "edges":       edges,
            "node_type":   node_type,
            "spatial_inform": spatial_inform,
            "conditions":  conditions,
            "dt":          dt,
            "max_start":   max_start,
        }

    # ------------------------------------------------------------------
    # Window loading (reads only active fields from HDF5)
    # ------------------------------------------------------------------

    def _load_window(self, path: str, indices: np.ndarray, start: int):
        with h5py.File(path, "r") as f:
            time_idx = start + np.arange(0, self.horizon + self.pf_extra + 1) * self.time_stride
            channels = []
            for fname in self.fields:
                data_all = f[f"state/{fname}"][time_idx]
                d = data_all[:, indices, 0]
                channels.append(d)
            state = np.stack(channels, axis=-1).astype(np.float32)  # [T+1, N, n_active]
            time_all = f["time"][time_idx]
        return state, time_all

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta  = self.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])

        # Load raw active-field data: [T+1, N, n_active]
        state_np, time_seq_np = self._load_window(path, meta["indices"], start_idx)

        # ---- Optional liquid_cut ----
        liquid_cut = self.config.get("liquid_cut", False)
        if liquid_cut:
            node_y = meta["node_pos"][:, 1]
            y_cutoff_mask = node_y > 1e-4 + 1e-6
            for local_i, fname in enumerate(self.fields):
                if fname == "gamma_liquid":
                    state_np[..., local_i][:, y_cutoff_mask] = 0.0

        # ---- Active mask for loss weighting ----
        active_mask = build_active_mask(state_np, self.fields, self.mask_cfg)

        # ---- Normalize active fields ----
        state = torch.from_numpy(state_np)          # [T+1, N, n_active]
        if self.normalize:
            state = (state - self.norm_mean) / self.norm_std
            node_pos_scaled = meta["node_pos_scaled"]
        else:
            node_pos_scaled = meta["node_pos"]

        # ---- Build slot tensor [T+1, N, K] ----
        T1, N, _ = state.shape
        slot_values = torch.zeros(T1, N, self.K, dtype=state.dtype)
        for local_i, fname in enumerate(self.fields):
            slot_idx = self.field_to_slot.get(fname)
            if slot_idx is not None:
                slot_values[..., slot_idx] = state[..., local_i]

        # Apply per-slot defaults for inactive slots (usually 0.0)
        for slot_idx, default_val in enumerate(self.slot_defaults):
            if slot_idx not in self.active_slot_indices and default_val != 0.0:
                slot_values[..., slot_idx] = default_val

        # ---- Presence mask and active indices ----
        presence_mask = torch.zeros(self.K, dtype=torch.float32)
        for si in self.active_slot_indices:
            presence_mask[si] = 1.0
        active_indices = torch.tensor(self.active_slot_indices, dtype=torch.long)

        # ---- Slot type indices ----
        slot_type_tensor = torch.tensor(self.slot_type_indices, dtype=torch.long)

        # ---- Time ----
        rel_time = time_seq_np[1:] - time_seq_np[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)  # [T, 1]

        sample = {
            # Primary slot-based outputs (for LPBF-NO v3)
            "slot_values":      slot_values,          # [T+1, N, K]
            "presence_mask":    presence_mask,         # [K]
            "active_indices":   active_indices,        # [n_active]
            "slot_types":       slot_type_tensor,      # [K]

            # Legacy state output (for backward compatibility)
            "state":            slot_values,           # alias, same tensor

            # Positional / structural
            "node_pos":         node_pos_scaled,       # [N, 3] normalized [0,1]
            "node_pos_physical":meta["node_pos"],      # [N, 3] physical coords (for laser)
            "edges":            meta["edges"],         # [ne, 2]
            "node_type":        meta["node_type"],     # [N, 1]
            "spatial_inform":   meta["spatial_inform"],# [10]

            # Temporal / condition
            "conditions":       meta["conditions"],    # [cond_dim]
            "time_seq":         time_tensor / self.time_ref,  # [T, 1]
            "dt":               meta["dt"] * self.time_stride / self.time_ref,

            # Grid shape (needed for gradient losses)
            "grid_shape":       torch.tensor(list(meta["ds_shape"])),
        }

        if active_mask is not None:
            # Expand active_mask to slot dimension if needed
            # active_mask is [T, N, n_active]; we expand to [T, N, K]
            T_h = active_mask.shape[0]
            slot_active_mask = torch.zeros(T_h, N, self.K, dtype=active_mask.dtype)
            for local_i in range(len(self.fields)):
                si = self.active_slot_indices[local_i]
                slot_active_mask[..., si] = active_mask[..., local_i]
            sample["active_mask"] = slot_active_mask

        return sample
