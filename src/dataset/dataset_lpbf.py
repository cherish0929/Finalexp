"""
dataset_lpbf.py — LPBF Dataset with Slot-based Design + Laser Physics

Two dataset classes:
  LPBFSlotDataset     — core slot-based dataset (field-agnostic fixed-K design)
  LPBFLaserDataset    — extends with laser trajectory and physical parameters

Slot design:
  - Predefined K field slots with presence_mask and field_type_embedding indices
  - Input dimension is fixed regardless of which fields are active in a given run
  - Fully backward-compatible: existing `state` field still returned

Coordinate convention:
  x = laser scan direction
  y = depth direction (air above, metal below)
  z = lateral (scan width) direction

Grid storage order in HDF5:
  for z: for y: for x:  →  index = z * Nx * Ny + y * Nx + x
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import ChannelNormalizer, build_active_mask
from src.dataset.dataset_fast import (
    _read_file_list,
    _normalize_stride,
    _compute_downsample_indices,
    _build_grid_edges,
    _build_node_type,
    _process_condition_normalize,
    _condition_vector,
)


# ---------------------------------------------------------------------------
# Slot metadata helpers
# ---------------------------------------------------------------------------

FIELD_TYPE_REGISTRY: Dict[str, str] = {
    "T":              "temperature",
    "Ux":             "velocity",
    "Uy":             "velocity",
    "Uz":             "velocity",
    "alpha.air":      "interface",
    "alpha.titanium": "interface",
    "gamma_liquid":   "interface",
}

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
    Slot-based LPBF dataset.

    Each __getitem__ returns:
      state / slot_values : [T+1, N, K]  slot-aligned state (normalized)
      presence_mask       : [K]           1.0 for active slots
      active_indices      : [n_active]    indices of active slots
      slot_types          : [K] (int)     0=temperature,1=interface,2=velocity,3=generic
      node_pos            : [N, 3]        normalized [0,1] node positions
      node_pos_physical   : [N, 3]        un-normalized physical coords
      edges               : [ne, 2]       grid edges
      node_type           : [N, 1]        boundary type
      spatial_inform      : [10]          grid/time metadata
      conditions          : [cond_dim]    normalized process parameters
      time_seq            : [T, 1]        relative time steps
      dt                  : float         time step size
      grid_shape          : [3]           (Nx, Ny, Nz)
    """

    SLOT_TYPE_TO_IDX: Dict[str, int] = {
        "temperature": 0,
        "interface":   1,
        "velocity":    2,
        "generic":     3,
    }

    def __init__(self, args, mode: str = "train", mat_data=None, spatial_stride=None):
        super().__init__()
        assert mode in {"train", "test"}
        data_cfg = args.data
        self.config = data_cfg
        self.mode = mode

        self.fields = data_cfg.get("fields", ["T"])

        model_cfg = args.model if hasattr(args, "model") else {}
        raw_slots = model_cfg.get("field_slots", None)
        if raw_slots is None:
            raw_slots = _build_default_slots(self.fields)

        self.slot_names: List[str] = [s["name"] for s in raw_slots]
        self.slot_types_str: List[str] = [
            s.get("type", FIELD_TYPE_REGISTRY.get(s["name"], "generic"))
            for s in raw_slots
        ]
        self.slot_defaults: List[float] = [float(s.get("default", 0.0)) for s in raw_slots]
        self.K = len(self.slot_names)

        self.slot_type_indices: List[int] = [
            self.SLOT_TYPE_TO_IDX.get(t, 3) for t in self.slot_types_str
        ]

        self.field_to_slot: Dict[str, int] = {
            name: i for i, name in enumerate(self.slot_names) if name in self.fields
        }
        self.active_slot_indices: List[int] = [
            i for i, name in enumerate(self.slot_names) if name in self.fields
        ]

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
        self.time_ref = 2e-5 if data_cfg.get("dt_scale", False) else 1
        self.edge_sample_ratio = data_cfg.get("edge_sample_ratio", 1.0)

        self.mask_cfg = args.train.get(
            "weight_loss",
            {"field": ["T", "alpha.air"], "threshold": [800, [0.4, 0.6]]}
        )

        self.file_paths = _read_file_list(data_cfg[f"{mode}_list"])
        self.meta_cache = {}
        self.sample_keys = []
        self.max_start_per_file = []

        valid_paths = []
        for path in self.file_paths:
            try:
                meta = self._build_meta(path)
            except OSError as e:
                print(f"[WARNING] 跳过损坏的 HDF5 文件: {path}\n  原因: {e}")
                continue

            file_id = len(valid_paths)
            valid_paths.append(path)
            self.meta_cache[path] = meta
            self.max_start_per_file.append(meta["max_start"])

            if mode == "train":
                for _ in range(self.samples_per_file):
                    self.sample_keys.append((file_id, None))
            else:
                step = max(1, self.horizon // 2)
                for start in range(1, meta["max_start"] + 1, step):
                    self.sample_keys.append((file_id, start))

        skipped = len(self.file_paths) - len(valid_paths)
        self.file_paths = valid_paths
        if skipped > 0:
            print(f"[WARNING] 共跳过 {skipped} 个损坏文件，"
                  f"有效文件 {len(valid_paths)}/{len(valid_paths) + skipped}")
        if len(valid_paths) == 0:
            raise RuntimeError("所有 HDF5 文件均损坏，无法创建数据集")

        example_meta = next(iter(self.meta_cache.values()))
        self.cond_dim = example_meta["conditions"].shape[-1]
        self.node_num = example_meta["node_pos"].shape[0]
        self.dt = example_meta["dt"] / self.time_ref

        if self.normalize:
            self.normalizer = self._load_normalizer()
        else:
            self.normalizer = ChannelNormalizer(
                np.zeros(len(self.fields), dtype=np.float32),
                np.ones(len(self.fields), dtype=np.float32),
            )

        self._sync_norm_cache()

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
        self.norm_mean = self.normalizer.mean
        self.norm_std = self.normalizer.std + self.normalizer.eps
        if self.normalize:
            for path, meta in self.meta_cache.items():
                meta["node_pos_scaled"] = self._scale_3D_pos(meta["node_pos"])

    @staticmethod
    def _scale_3D_pos(node_pos: torch.Tensor) -> torch.Tensor:
        pos_min = node_pos.min(dim=0).values
        pos_max = node_pos.max(dim=0).values
        return (node_pos - pos_min) / (pos_max - pos_min + 1e-8)

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
            )

            point_all = f["point"][:]
            point = point_all[indices]
            node_pos = torch.from_numpy(point.astype(np.float32))

            edges = _build_grid_edges(ds_shape, self.edge_sample_ratio)
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
            "grid_shape": grid_shape,
            "indices": indices,
            "ds_shape": ds_shape,
            "node_pos": node_pos,
            "edges": edges,
            "node_type": node_type,
            "spatial_inform": spatial_inform,
            "conditions": conditions,
            "dt": dt,
            "max_start": max_start,
        }

    def _load_window(self, path: str, indices: np.ndarray, start: int):
        with h5py.File(path, "r") as f:
            time_idx = start + np.arange(0, self.horizon + self.pf_extra + 1) * self.time_stride
            channels = []
            for fname in self.fields:
                data_all = f[f"state/{fname}"][time_idx]
                d = data_all[:, indices, 0]
                channels.append(d)
            state = np.stack(channels, axis=-1).astype(np.float32)
            time_all = f["time"][time_idx]
        return state, time_all

    def __len__(self):
        return len(self.sample_keys)

    def _resolve_start(self, idx: int):
        file_id, start_idx = self.sample_keys[idx]
        path = self.file_paths[file_id]
        meta = self.meta_cache[path]
        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])
        return file_id, path, meta, start_idx

    def __getitem__(self, idx):
        file_id, path, meta, start_idx = self._resolve_start(idx)

        state_np, time_seq_np = self._load_window(path, meta["indices"], start_idx)

        liquid_cut = self.config.get("liquid_cut", False)
        if liquid_cut:
            node_y = meta["node_pos"][:, 1]
            y_cutoff_mask = node_y > 1e-4 + 1e-6
            for local_i, fname in enumerate(self.fields):
                if fname == "gamma_liquid":
                    state_np[..., local_i][:, y_cutoff_mask] = 0.0

        active_mask = build_active_mask(state_np, self.fields, self.mask_cfg)

        state = torch.from_numpy(state_np)
        if self.normalize:
            state = (state - self.norm_mean) / self.norm_std
            node_pos_scaled = meta["node_pos_scaled"]
        else:
            node_pos_scaled = meta["node_pos"]

        T1, N, _ = state.shape
        slot_values = torch.zeros(T1, N, self.K, dtype=state.dtype)
        for local_i, fname in enumerate(self.fields):
            slot_idx = self.field_to_slot.get(fname)
            if slot_idx is not None:
                slot_values[..., slot_idx] = state[..., local_i]

        for slot_idx, default_val in enumerate(self.slot_defaults):
            if slot_idx not in self.active_slot_indices and default_val != 0.0:
                slot_values[..., slot_idx] = default_val

        presence_mask = torch.zeros(self.K, dtype=torch.float32)
        for si in self.active_slot_indices:
            presence_mask[si] = 1.0
        active_indices = torch.tensor(self.active_slot_indices, dtype=torch.long)
        slot_type_tensor = torch.tensor(self.slot_type_indices, dtype=torch.long)

        rel_time = time_seq_np[1:] - time_seq_np[0]
        time_tensor = torch.from_numpy(rel_time.astype(np.float32)).unsqueeze(-1)

        sample = {
            "slot_values":       slot_values,
            "presence_mask":     presence_mask,
            "active_indices":    active_indices,
            "slot_types":        slot_type_tensor,
            "state":             slot_values,
            "node_pos":          node_pos_scaled,
            "node_pos_physical": meta["node_pos"],
            "edges":             meta["edges"],
            "node_type":         meta["node_type"],
            "spatial_inform":    meta["spatial_inform"],
            "conditions":        meta["conditions"],
            "time_seq":          time_tensor / self.time_ref,
            "dt":                meta["dt"] * self.time_stride / self.time_ref,
            "grid_shape":        torch.tensor(list(meta["ds_shape"])),
        }

        if active_mask is not None:
            T_h = active_mask.shape[0]
            slot_active_mask = torch.zeros(T_h, N, self.K, dtype=active_mask.dtype)
            for local_i in range(len(self.fields)):
                si = self.active_slot_indices[local_i]
                slot_active_mask[..., si] = active_mask[..., local_i]
            sample["active_mask"] = slot_active_mask

        return sample


# ---------------------------------------------------------------------------
# Laser trajectory helpers
# ---------------------------------------------------------------------------

def _extract_laser_params(f: h5py.File) -> np.ndarray:
    """Extract raw laser physical parameters from HDF5.

    Returns float32 array [4]: [P_L (W), r (m), absorptivity, V_scan (m/s)]
    """
    thermal = f["parameter/thermal"][:].reshape(-1)
    return np.array(
        [thermal[3], thermal[4], thermal[7], thermal[8]],
        dtype=np.float32,
    )


def _build_laser_trajectory(f: h5py.File, time_all: np.ndarray) -> np.ndarray:
    """Interpolate dump laser positions to simulation timesteps.

    Returns float32 array [len(time_all), 3].
    """
    dump = f["parameter/dump"][:]
    n_dump = dump.shape[0]
    t0, t1 = float(time_all[0]), float(time_all[-1])
    dump_times = np.linspace(t0, t1, n_dump)

    laser_x = np.interp(time_all, dump_times, dump[:, 0]).astype(np.float32)
    laser_y = np.interp(time_all, dump_times, dump[:, 1]).astype(np.float32)
    laser_z = np.interp(time_all, dump_times, dump[:, 2]).astype(np.float32)

    return np.stack([laser_x, laser_y, laser_z], axis=-1)


# ---------------------------------------------------------------------------
# LPBFLaserDataset
# ---------------------------------------------------------------------------

class LPBFLaserDataset(LPBFSlotDataset):
    """
    Extends LPBFSlotDataset with laser-physics metadata.

    Additional __getitem__ keys:
        node_pos_abs    [N, 3]   physical node coordinates (m)
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
            meta["laser_traj"] = torch.from_numpy(_build_laser_trajectory(f, time_all))
            meta["time_all"] = time_all
        return meta

    def __getitem__(self, idx: int):
        file_id, path, meta, start_idx = self._resolve_start(idx)

        # Set resolved start so parent __getitem__ uses same index
        original = self.sample_keys[idx]
        self.sample_keys[idx] = (file_id, start_idx)
        sample = super().__getitem__(idx)
        self.sample_keys[idx] = original

        time_idx = start_idx + np.arange(
            0, self.horizon + self.pf_extra + 1
        ) * self.time_stride

        abs_time = meta["time_all"][time_idx]
        sample["abs_time_seq"] = torch.from_numpy(abs_time)
        sample["laser_traj"] = meta["laser_traj"][time_idx]
        sample["laser_params"] = meta["laser_params"]
        sample["node_pos_abs"] = meta["node_pos"]

        return sample
