"""
dataset_lpbf.py — LPBF Dataset with Laser Physics Data

Extends LPBFSlotDataset to also expose:
  - node_pos_abs:  un-normalized absolute node positions [N, 3] (meters)
  - laser_params:  raw physical laser params [4]  (P_L, r, V_scan, absorptivity)
  - laser_traj:    interpolated laser xyz at each timestep [T_total, 3] (meters)
  - abs_time_seq:  absolute timestamps for the window [T+1] (seconds)

These additional fields feed the LaserFieldModule in physgto_lpbf.py.

All other behaviour (slot values, presence_mask, normalisation, etc.) is identical
to LPBFSlotDataset.  Existing configs that do not use this dataset are unaffected.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.dataset_fast import (
    _read_file_list,
    _normalize_stride,
    _compute_downsample_indices,
    _build_grid_edges,
    _build_node_type,
    _process_condition_normalize,
    _condition_vector,
)
from src.dataset.dataset_lpbf_v3 import (
    LPBFSlotDataset,
    FIELD_STATS_DEFAULT,
    _build_default_slots,
    FIELD_TYPE_REGISTRY,
)
from src.utils import ChannelNormalizer, build_active_mask


def _extract_laser_params(f: h5py.File) -> np.ndarray:
    """Extract raw (un-normalised) laser physical parameters from HDF5.

    Indices mirror _process_condition_normalize in dataset_fast.py:
      thermal[3] = P_L        (W)
      thermal[4] = r          (m)
      thermal[7] = absorptivity
      thermal[8] = V_scan     (m/s)  ← thermal[5] is NOT scan speed

    Returns float32 array of shape [4].
    """
    thermal = f["parameter/thermal"][:].reshape(-1)
    return np.array(
        [thermal[3], thermal[4], thermal[7], thermal[8]],
        dtype=np.float32,
    )  # [P_L, r, absorptivity, V_scan]


def _build_laser_trajectory(f: h5py.File, time_all: np.ndarray) -> np.ndarray:
    """Interpolate dump laser positions to simulation timesteps.

    parameter/dump has shape [1200, 5]:
      col 0: x_laser (m)
      col 1: y_laser (m)
      col 2: z_laser (m)
      col 3: z_layer indicator (m)
      col 4: laser_on flag

    Returns float32 array of shape [len(time_all), 3].
    """
    dump = f["parameter/dump"][:]          # [1200, 5]
    n_dump = dump.shape[0]
    # Assume dump rows are uniformly spaced over the simulation time span
    t0, t1 = float(time_all[0]), float(time_all[-1])
    dump_times = np.linspace(t0, t1, n_dump)

    laser_x = np.interp(time_all, dump_times, dump[:, 0]).astype(np.float32)
    laser_y = np.interp(time_all, dump_times, dump[:, 1]).astype(np.float32)
    laser_z = np.interp(time_all, dump_times, dump[:, 2]).astype(np.float32)

    return np.stack([laser_x, laser_y, laser_z], axis=-1)  # [T_total, 3]


class LPBFLaserDataset(LPBFSlotDataset):
    """
    Extends LPBFSlotDataset with laser-physics metadata needed by physgto_lpbf.

    Additional __getitem__ keys:
        node_pos_abs    [N, 3]   physical (un-normalised) node coordinates (m)
        laser_params    [4]      [P_L (W), r (m), absorptivity, V_scan (m/s)]
        laser_traj      [T+1, 3] laser xyz at each window timestep (m)
        abs_time_seq    [T+1]    absolute timestamps for the window (s)
    """

    def _build_meta(self, path: str) -> dict:
        meta = super()._build_meta(path)

        path = str(Path(path).expanduser().resolve())
        with h5py.File(path, "r") as f:
            time_all = f["time"][:].astype(np.float32)
            meta["laser_params"] = torch.from_numpy(_extract_laser_params(f))
            meta["laser_traj"]   = torch.from_numpy(_build_laser_trajectory(f, time_all))
            meta["time_all"]     = time_all

        return meta

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        if start_idx is None:
            # super() already used random start; re-derive it from time_seq
            # The relative time_seq starts at 0 → absolute offset is start_idx.
            # Since super() randomised start_idx internally, we infer via time_seq.
            # Instead, we need the actual start_idx. Override: store it here.
            # This requires a small hack: peek at what super chose.
            # Simpler: use _resolve_start() pattern below.
            pass

        # Re-read start_idx the same way super() did (random selection is not stored).
        # We work around this by storing the resolved start on the sample dict itself
        # inside a patched __getitem__ that calls our own _resolve_and_get.
        return sample  # laser keys added by _patch_sample below (see __getitem__ below)


# LPBFLaserDataset needs a clean override of __getitem__ that resolves start_idx
# once, passes it to both the slot logic and the laser-traj slice.

class LPBFLaserDataset(LPBFSlotDataset):
    """
    Extends LPBFSlotDataset with laser-physics metadata needed by physgto_lpbf.

    Additional __getitem__ keys:
        node_pos_abs    [N, 3]   physical (un-normalised) node coordinates (m)
        laser_params    [4]      [P_L (W), r (m), absorptivity, V_scan (m/s)]
        laser_traj      [T+1, 3] laser xyz at each window timestep (m)
        abs_time_seq    [T+1]    absolute timestamps for the window (s)
    """

    def _build_meta(self, path: str) -> dict:
        meta = super()._build_meta(path)

        path_resolved = str(Path(path).expanduser().resolve())
        with h5py.File(path_resolved, "r") as f:
            time_all = f["time"][:].astype(np.float32)
            meta["laser_params"] = torch.from_numpy(_extract_laser_params(f))
            meta["laser_traj"]   = torch.from_numpy(_build_laser_trajectory(f, time_all))
            meta["time_all"]     = time_all  # [T_total]

        return meta

    def __getitem__(self, idx: int):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]

        # Resolve random start index deterministically here (once) so we can
        # use the same index for both the slot state and the laser trajectory slice.
        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])
            # Temporarily patch sample_keys so super().__getitem__ uses the same start
            original = self.sample_keys[idx]
            self.sample_keys[idx] = (file_id, start_idx)
            sample = super().__getitem__(idx)
            self.sample_keys[idx] = original
        else:
            sample = super().__getitem__(idx)

        # Compute time indices for this window
        time_idx = start_idx + np.arange(
            0, self.horizon + self.pf_extra + 1
        ) * self.time_stride  # [T+1]

        # Absolute time sequence
        abs_time = meta["time_all"][time_idx]               # [T+1]
        sample["abs_time_seq"] = torch.from_numpy(abs_time) # [T+1]

        # Laser trajectory for this window
        laser_traj_window = meta["laser_traj"][time_idx]    # [T+1, 3]
        sample["laser_traj"] = laser_traj_window            # already a tensor

        # Raw laser params (shared for all timesteps in this file)
        sample["laser_params"] = meta["laser_params"]       # [4]

        # Physical (un-normalised) node positions
        sample["node_pos_abs"] = meta["node_pos"]           # [N, 3]

        return sample
