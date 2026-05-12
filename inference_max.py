# -*- coding: utf-8 -*-
"""
inference_max.py
================

统一版推理脚本：
1. 支持单个 config 或多个 config 依次推理
2. 推理结果默认保存到 <save_path>/inference，也可手动指定目录名
3. 非体积分数字段沿用 inference_v1 的可视化逻辑
4. 体积分数字段沿用 inference_air 的界面指标与可视化逻辑
5. 支持 sample_idxs 一次推理多个样本
6. 每次推理后自动生成带时间戳的指标汇总 txt
7. 生成 GIF 时可叠加每一步误差指标
8. 对 3D / cut / 2D / LPBF 数据分支统一兼容

默认使用方式：
1. 直接修改 DEFAULT_INFERENCE_CONFIG
2. 运行 `python3 inference_max.py`

如需保留命令行方式，可手动调用 `main_cli()`
"""

import argparse
import io
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from torch.amp import autocast
from tqdm import tqdm

from src.dataset import AeroGtoDataset, AeroGtoDataset2D, CutAeroGtoDataset, LPBFLaserDataset
from src.model import build_model
from src.train import (
    _autoregressive_lpbf,
    _each_l2,
    _masked_relative_l2,
    _masked_rmse,
    _relative_l2,
    _rmse,
)
from src.utils import load_json_config, set_seed


def _is_vof_field(field_name: str) -> bool:
    name = field_name.lower()
    return any(key in name for key in ("alpha", "gamma", "frac"))


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(".", "_").replace(" ", "_")


def _fmt_scalar(val, sci: bool = True) -> str:
    if val is None:
        return "None"
    try:
        fval = float(val)
    except Exception:
        return str(val)
    if np.isnan(fval):
        return "nan"
    if sci:
        return f"{fval:.4e}"
    return f"{fval:.4f}"


def _fmt_array(arr, sci: bool = True) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return _fmt_scalar(arr.item(), sci=sci)
    if sci:
        return "[" + ", ".join(f"{v:.4e}" if np.isfinite(v) else "nan" for v in arr) + "]"
    return "[" + ", ".join(f"{v:.4f}" if np.isfinite(v) else "nan" for v in arr) + "]"


def _each_step_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    t_step = target.shape[1]
    values = torch.zeros(t_step, device=pred.device)
    for t in range(t_step):
        diff = pred[:, t] - target[:, t]
        values[t] = torch.sqrt(torch.mean(diff ** 2))
    return values


def _interface_iou(pred_np, gt_np, threshold=0.5):
    pred_bin = (pred_np >= threshold).astype(bool)
    gt_bin = (gt_np >= threshold).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return intersection / union


def _interface_dice(pred_np, gt_np, threshold=0.5):
    pred_bin = (pred_np >= threshold).astype(bool)
    gt_bin = (gt_np >= threshold).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def _interface_band_mae(pred_np, gt_np, band_lo=0.2, band_hi=0.8):
    mask = (gt_np >= band_lo) & (gt_np <= band_hi)
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(pred_np[mask] - gt_np[mask]))


class InferenceMaxPredictor:
    def __init__(self, config_path, mode="test", model_path=None, device_str=None):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.args = load_json_config(self.config_path)

        if device_str is None:
            device_str = getattr(self.args, "device", "cuda")
        if "cuda" in str(device_str) and not torch.cuda.is_available():
            device_str = "cpu"

        self.device = torch.device(device_str)
        self.mode = mode
        self.model_name = self.args.model.get("name", "PhysGTO")
        self.use_amp = bool(self.args.train.get("use_amp", False))
        self.check_point = self.args.train.get("check_point", False)

        print(f"[Init] Config: {self.config_path}")
        print(f"[Init] Device: {self.device}")

        self.dataset_class, self.dataset_kind = self._select_dataset_class()
        self.train_dataset = self._build_dataset(mode="train")
        if mode == "test":
            self.dataset = self._build_dataset(mode="test", train_dataset=self.train_dataset)
        else:
            self.dataset = self.train_dataset

        self.fields = list(self.dataset.fields)
        self.vof_fields = [(fname, i) for i, fname in enumerate(self.fields) if _is_vof_field(fname)]

        print(f"[Init] Dataset kind: {self.dataset_kind}")
        print(f"[Init] Fields: {self.fields}")
        if self.vof_fields:
            print(f"[Init] VOF fields: {[name for name, _ in self.vof_fields]}")
        else:
            print("[Init] No VOF field found in predicted fields.")

        cond_dim = self.args.model.get("cond_dim") or self.train_dataset.cond_dim
        default_dt = self.args.model.get("dt", self.train_dataset.dt)

        if self.model_name == "gto_lpbf":
            self.args.model["_fields"] = self.args.data.get("fields", ["T"])

        self.model = build_model(self.args.model, cond_dim, default_dt, self.device)
        self.model.eval()

        if model_path is None:
            save_root = Path(self.args.save_path)
            model_path = save_root / "nn" / f"{self.args.name}_best.pt"
        self.model_path = str(Path(model_path).expanduser().resolve())
        self._load_weights(self.model_path)

        self.normalizer = self.dataset.normalizer
        self.normalizer.to(self.device)

    def _select_dataset_class(self):
        data_cfg = self.args.data
        model_cfg = self.args.model
        space_dim = model_cfg.get("space_size", 3)

        if self.model_name == "gto_lpbf":
            return LPBFLaserDataset, "lpbf"
        if space_dim == 3:
            if data_cfg.get("cut", False):
                return CutAeroGtoDataset, "cut"
            return AeroGtoDataset, "3d"
        return AeroGtoDataset2D, "2d"

    def _build_dataset(self, mode, train_dataset=None):
        if self.dataset_kind in ("3d", "cut", "lpbf"):
            mat_data = None
            if train_dataset is not None and getattr(train_dataset, "normalize", False):
                mat_data = train_dataset.mat_mean_and_std
            dataset = self.dataset_class(args=self.args, mode=mode, mat_data=mat_data)
            if train_dataset is not None:
                dataset.normalizer = train_dataset.normalizer
                if hasattr(dataset, "_sync_norm_cache"):
                    dataset._sync_norm_cache()
            return dataset

        data_cfg = self.args.data
        mat_data = None
        if train_dataset is not None and getattr(train_dataset, "normalize", False):
            mat_data = train_dataset.mat_mean_and_std

        dataset = self.dataset_class(
            file_list=data_cfg[f"{mode}_list"],
            mode=mode,
            fields=data_cfg.get("fields", ["T"]),
            input_steps=data_cfg.get("input_steps", 1),
            horizon=data_cfg.get(f"horizon_{mode}", 1),
            time_stride=data_cfg.get("time_stride", 1),
            spatial_stride=data_cfg.get("spatial_stride", 1),
            normalize=data_cfg.get("normalize", True),
            samples_per_file=data_cfg.get("samples_per_file", 32),
            norm_cache=data_cfg.get("norm_cache"),
            slice_z=data_cfg.get("slice_z", 5e-4),
            mat_data=mat_data,
        )
        if train_dataset is not None:
            dataset.normalizer = train_dataset.normalizer
        return dataset

    def _load_weights(self, model_path):
        print(f"[Init] Loading weights: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weight file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "ema_shadow" in checkpoint:
            print("[Init] Using EMA shadow weights.")
            ema_shadow = checkpoint["ema_shadow"]
            state_dict = self.model.state_dict()
            for name, tensor in ema_shadow.items():
                if name in state_dict:
                    state_dict[name] = tensor
            self.model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)

    def _resolve_sample(self, sample_idx):
        file_id, start_idx = self.dataset.sample_keys[sample_idx]
        path = self.dataset.file_paths[file_id]
        meta = self.dataset.meta_cache[path]

        if start_idx is None:
            start_idx = random.randint(1, meta["max_start"])
            original = self.dataset.sample_keys[sample_idx]
            self.dataset.sample_keys[sample_idx] = (file_id, start_idx)
            try:
                sample = self.dataset[sample_idx]
            finally:
                self.dataset.sample_keys[sample_idx] = original
        else:
            sample = self.dataset[sample_idx]

        return sample, file_id, path, meta, start_idx

    def _to_device_scalar_dt(self, dt):
        if torch.is_tensor(dt):
            dt_t = dt.to(self.device, dtype=torch.float32)
            if dt_t.ndim == 0:
                dt_t = dt_t.view(1)
            return dt_t
        return torch.tensor([float(dt)], device=self.device, dtype=torch.float32)

    def _recover_coords(self, sample, meta):
        if "node_pos_abs" in sample:
            coords = sample["node_pos_abs"]
            if torch.is_tensor(coords):
                return coords.cpu().numpy()
            return np.asarray(coords)

        if self.dataset_kind == "cut" and "spatial_inform" in sample:
            spatial = sample["spatial_inform"]
            if torch.is_tensor(spatial):
                spatial = spatial.cpu().numpy()
            node_pos = sample["node_pos"]
            if torch.is_tensor(node_pos):
                node_pos = node_pos.cpu().numpy()
            pos_min = np.array([spatial[0], spatial[2], spatial[4]], dtype=np.float32)
            pos_max = np.array([spatial[1], spatial[3], spatial[5]], dtype=np.float32)
            return node_pos * (pos_max - pos_min + 1e-8) + pos_min

        if "node_pos" in meta:
            coords = meta["node_pos"]
            if torch.is_tensor(coords):
                return coords.cpu().numpy()
            return np.asarray(coords)

        node_pos = sample["node_pos"]
        if torch.is_tensor(node_pos):
            return node_pos.cpu().numpy()
        return np.asarray(node_pos)

    def _load_external_field_window(self, path, meta, start_idx, field_name):
        time_idx = start_idx + np.arange(0, self.dataset.horizon + 1) * self.dataset.time_stride

        try:
            with h5py.File(path, "r") as f:
                key = f"state/{field_name}"
                if key not in f:
                    return None

                if self.dataset_kind != "cut":
                    indices = meta["indices"]
                    data = f[key][time_idx[1:]][:, indices, 0]
                    return data

                nx, ny, nz = meta["ds_shape"]
                indices = meta["indices"]

                gamma_3d = None
                if "state/gamma_liquid" in f:
                    gamma_3d = f["state/gamma_liquid"][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)

                air_3d = None
                if "state/alpha.air" in f:
                    air_3d = f["state/alpha.air"][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)

                x_min, x_max, y_min, y_max, z_min, z_max = self.dataset._compute_bounds(
                    gamma_3d, air_3d, nx, ny, nz
                )
                data_3d = f[key][time_idx][:, indices, 0].reshape(-1, nz, ny, nx)
                crop = data_3d[:, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
                return crop.reshape(crop.shape[0], -1)[1:]
        except Exception as exc:
            print(f"[Warn] Failed to load external field '{field_name}' from {path}: {exc}")
            return None

    def _build_interface_sequences(self, path, meta, start_idx, pred_real, gt_real, interface_field):
        gt_interface = None
        pred_interface = None

        if interface_field is None:
            return gt_interface, pred_interface

        if interface_field in self.fields:
            idx = self.fields.index(interface_field)
            gt_interface = gt_real[0, :, :, idx].cpu().numpy()
            pred_interface = pred_real[0, :, :, idx].cpu().numpy()
            return gt_interface, pred_interface

        gt_interface = self._load_external_field_window(path, meta, start_idx, interface_field)
        return gt_interface, pred_interface

    def _run_autoregressive(self, sample):
        state_seq = sample["state"].unsqueeze(0).to(self.device)
        node_pos = sample["node_pos"].unsqueeze(0).to(self.device)
        edges = sample["edges"].unsqueeze(0).to(self.device)
        time_seq = sample["time_seq"].unsqueeze(0).to(self.device)
        conditions = sample["conditions"].unsqueeze(0).to(self.device).float()
        dt = self._to_device_scalar_dt(sample["dt"])

        batch_like = {}
        if "spatial_inform" in sample:
            batch_like["spatial_inform"] = sample["spatial_inform"].unsqueeze(0).to(self.device)
        if "node_pos_abs" in sample:
            batch_like["node_pos_abs"] = sample["node_pos_abs"].unsqueeze(0).to(self.device)
        if "laser_params" in sample:
            batch_like["laser_params"] = sample["laser_params"].unsqueeze(0).to(self.device).float()
        if "laser_traj" in sample:
            batch_like["laser_traj"] = sample["laser_traj"].unsqueeze(0).to(self.device)
        if "abs_time_seq" in sample:
            batch_like["abs_time_seq"] = sample["abs_time_seq"].unsqueeze(0).to(self.device)

        active_mask = sample.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask.unsqueeze(0).to(self.device)[:, 1:]

        state_0 = state_seq[:, 0]
        gt_seq = state_seq[:, 1:]

        use_cuda_amp = self.use_amp and self.device.type == "cuda"
        with torch.no_grad():
            if use_cuda_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred_seq = self._call_model(state_0, node_pos, edges, time_seq, conditions, dt, batch_like)
            else:
                pred_seq = self._call_model(state_0, node_pos, edges, time_seq, conditions, dt, batch_like)

            pred_real = self.normalizer.denormalize(pred_seq.float())
            gt_real = self.normalizer.denormalize(gt_seq.float())

        for _, idx in self.vof_fields:
            pred_real[..., idx] = pred_real[..., idx].clamp(0.0, 1.0)
            gt_real[..., idx] = gt_real[..., idx].clamp(0.0, 1.0)

        return pred_seq.float(), gt_seq.float(), pred_real.float(), gt_real.float(), active_mask

    def _call_model(self, state_0, node_pos, edges, time_seq, conditions, dt, batch_like):
        use_spatial = self.model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max", "gto_lpbf")
        if self.model_name == "gto_lpbf":
            return _autoregressive_lpbf(
                self.model,
                state_0,
                node_pos,
                edges,
                time_seq,
                batch_like["spatial_inform"],
                conditions,
                dt,
                self.check_point,
                batch_like,
                self.device,
            )
        if use_spatial:
            return self.model.autoregressive(
                state_0,
                node_pos,
                edges,
                time_seq,
                batch_like["spatial_inform"],
                conditions,
                dt,
                check_point=self.check_point,
            )
        return self.model.autoregressive(
            state_0,
            node_pos,
            edges,
            time_seq,
            conditions,
            dt,
            check_point=self.check_point,
        )

    def compute_metrics(self, pred_seq, gt_seq, pred_real, gt_real, active_mask, sample_idx):
        metrics = {
            "sample_idx": sample_idx,
            "MSE_normalized": torch.mean((pred_seq - gt_seq) ** 2).item(),
            "mean_L2_all": _relative_l2(pred_real, gt_real).item(),
            "each_step_L2_all": _each_l2(pred_real, gt_real).cpu().numpy(),
            "each_step_RMSE_all": _each_step_rmse(pred_real, gt_real).cpu().numpy(),
            "per_field": {},
        }

        if active_mask is not None:
            inactive_mask = ~active_mask
            active_vals = []
            inactive_vals = []
        else:
            inactive_mask = None
            active_vals = None
            inactive_vals = None

        for idx, fname in enumerate(self.fields):
            pred_ch_real = pred_real[..., idx:idx + 1]
            gt_ch_real = gt_real[..., idx:idx + 1]
            rmse_val = _rmse(pred_ch_real, gt_ch_real)

            fm = {
                "field_type": "vof" if _is_vof_field(fname) else "generic",
                "relative_L2": _relative_l2(pred_ch_real, gt_ch_real).item(),
                "RMSE": float(rmse_val.reshape(-1)[0].item()),
                "each_step_L2": _each_l2(pred_ch_real, gt_ch_real).cpu().numpy(),
                "each_step_RMSE": _each_step_rmse(pred_ch_real, gt_ch_real).cpu().numpy(),
            }

            if active_mask is not None:
                ch_active = active_mask[..., idx:idx + 1]
                ch_inactive = inactive_mask[..., idx:idx + 1]
                fm["active_L2"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_active)
                fm["inactive_L2"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_inactive)
                fm["active_RMSE"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_active)
                fm["inactive_RMSE"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_inactive)
                if not np.isnan(fm["active_L2"]):
                    active_vals.append(fm["active_L2"])
                if not np.isnan(fm["inactive_L2"]):
                    inactive_vals.append(fm["inactive_L2"])

            if _is_vof_field(fname):
                pred_np = pred_real[0, :, :, idx].cpu().numpy()
                gt_np = gt_real[0, :, :, idx].cpu().numpy()
                ious, dices, band_maes = [], [], []
                for t in range(pred_np.shape[0]):
                    ious.append(_interface_iou(pred_np[t], gt_np[t]))
                    dices.append(_interface_dice(pred_np[t], gt_np[t]))
                    band_maes.append(_interface_band_mae(pred_np[t], gt_np[t]))
                fm["IoU_per_step"] = np.asarray(ious, dtype=np.float64)
                fm["Dice_per_step"] = np.asarray(dices, dtype=np.float64)
                fm["band_MAE_per_step"] = np.asarray(band_maes, dtype=np.float64)
                fm["mean_IoU"] = float(np.mean(ious))
                fm["mean_Dice"] = float(np.mean(dices))
                fm["mean_band_MAE"] = float(np.mean(band_maes))

            metrics["per_field"][fname] = fm

        if active_mask is not None:
            metrics["active_mean_L2"] = float(np.mean(active_vals)) if active_vals else float("nan")
            metrics["inactive_mean_L2"] = float(np.mean(inactive_vals)) if inactive_vals else float("nan")

        return metrics

    def predict_and_evaluate(self, sample_idx, interface_field=None):
        sample, _, path, meta, start_idx = self._resolve_sample(sample_idx)
        pred_seq, gt_seq, pred_real, gt_real, active_mask = self._run_autoregressive(sample)
        coords = self._recover_coords(sample, meta)
        gt_interface, pred_interface = self._build_interface_sequences(
            path, meta, start_idx, pred_real, gt_real, interface_field
        )

        metrics = self.compute_metrics(pred_seq, gt_seq, pred_real, gt_real, active_mask, sample_idx)
        result = {
            "pred": pred_real[0].cpu().numpy(),
            "gt": gt_real[0].cpu().numpy(),
            "coords": coords,
            "gt_interface": gt_interface,
            "pred_interface": pred_interface,
        }
        return result, metrics

    def _build_slice_mask(self, coords, axis, slice_pos=None, min_points=32):
        coords = np.asarray(coords)
        if coords.shape[1] == 2:
            mask = np.ones(coords.shape[0], dtype=bool)
            return mask, slice_pos

        axis_id = {"x": 0, "y": 1, "z": 2}[axis]
        axis_values = coords[:, axis_id]
        if slice_pos is None:
            slice_pos = 0.5 * (np.nanmin(axis_values) + np.nanmax(axis_values))

        axis_span = max(np.nanmax(axis_values) - np.nanmin(axis_values), 1e-12)
        thickness = max(axis_span * 2e-3, 1e-8)
        mask = np.abs(axis_values - slice_pos) <= thickness

        for _ in range(8):
            if np.sum(mask) >= min_points:
                break
            thickness *= 1.8
            mask = np.abs(axis_values - slice_pos) <= thickness

        if np.sum(mask) < min_points:
            dist = np.abs(axis_values - slice_pos)
            k = min(len(dist), max(min_points, int(0.03 * len(dist))))
            idx = np.argpartition(dist, k - 1)[:k]
            mask = np.zeros_like(dist, dtype=bool)
            mask[idx] = True

        return mask, slice_pos

    def _project_slice_points(self, coords, axis="z", slice_pos=None, min_points=32):
        mask, slice_pos = self._build_slice_mask(coords, axis, slice_pos=slice_pos, min_points=min_points)

        if coords.shape[1] == 2:
            pts_x, pts_y = coords[mask, 0], coords[mask, 1]
            xlabel, ylabel = "X (m)", "Y (m)"
            return mask, slice_pos, pts_x, pts_y, xlabel, ylabel

        if axis == "x":
            pts_x, pts_y = coords[mask, 1], coords[mask, 2]
            xlabel, ylabel = "Y (m)", "Z (m)"
        elif axis == "y":
            pts_x, pts_y = coords[mask, 0], coords[mask, 2]
            xlabel, ylabel = "X (m)", "Z (m)"
        else:
            pts_x, pts_y = coords[mask, 0], coords[mask, 1]
            xlabel, ylabel = "X (m)", "Y (m)"

        return mask, slice_pos, pts_x, pts_y, xlabel, ylabel

    def _interp_grid(self, pts_x, pts_y, vals, Xi, Yi, method="cubic"):
        if vals is None:
            return None

        points = np.column_stack([pts_x, pts_y])
        values = np.asarray(vals)

        methods = [method]
        if method != "linear":
            methods.append("linear")
        methods.append("nearest")

        Z = None
        for current_method in methods:
            try:
                Z = griddata(points, values, (Xi, Yi), method=current_method)
            except Exception:
                Z = None
            if Z is not None and not np.all(np.isnan(Z)):
                break

        if Z is None:
            return None

        if np.isnan(Z).any():
            try:
                z_near = griddata(points, values, (Xi, Yi), method="nearest")
                Z = np.where(np.isnan(Z), z_near, Z)
            except Exception:
                pass

        return Z

    def _smooth(self, Z, sigma):
        if Z is None or sigma is None or sigma <= 0:
            return Z
        if np.all(np.isnan(Z)):
            return Z
        nan_mask = np.isnan(Z)
        fill_value = np.nanmedian(Z)
        if not np.isfinite(fill_value):
            fill_value = 0.0
        Z_fill = np.where(nan_mask, fill_value, Z)
        Z_smooth = gaussian_filter(Z_fill, sigma=sigma, mode="nearest")
        Z_smooth[nan_mask] = np.nan
        return Z_smooth

    def _render_array(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        buf.close()
        return img

    def _field_limits(self, result_dict, field_name, field_idx):
        fixed = {
            "T": (300, 3500),
            "alpha.air": (0, 1),
            "alpha.titanium": (0, 1),
            "gamma_liquid": (0, 1),
        }
        if field_name in fixed:
            return fixed[field_name]

        all_pred = result_dict["pred"][..., field_idx]
        all_gt = result_dict["gt"][..., field_idx]
        combined = np.concatenate([all_pred.reshape(-1), all_gt.reshape(-1)])
        vmin = np.nanpercentile(combined, 1.0)
        vmax = np.nanpercentile(combined, 99.0)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin = float(np.nanmin(combined))
            vmax = float(np.nanmax(combined))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        return vmin, vmax

    def _generic_metrics_text(self, metrics, field_name, time_step):
        fm = metrics["per_field"][field_name]
        line_step = (
            f"Step {time_step}   "
            f"L2(field): {fm['each_step_L2'][time_step]:.4e}    "
            f"RMSE(field): {fm['each_step_RMSE'][time_step]:.4e}    "
            f"L2(all): {metrics['each_step_L2_all'][time_step]:.4e}"
        )

        parts = [
            f"Overall   MSE(norm): {metrics['MSE_normalized']:.4e}",
            f"Rel L2({field_name}): {fm['relative_L2']:.4e}",
            f"RMSE({field_name}): {fm['RMSE']:.4e}",
        ]
        if "active_L2" in fm:
            parts.append(f"Active L2: {fm['active_L2']:.4e}" if not np.isnan(fm["active_L2"]) else "Active L2: nan")
            parts.append(
                f"Inactive L2: {fm['inactive_L2']:.4e}" if not np.isnan(fm["inactive_L2"]) else "Inactive L2: nan"
            )
        return line_step, "    ".join(parts)

    def _vof_metrics_text(self, metrics, field_name, time_step):
        fm = metrics["per_field"][field_name]
        line_step = (
            f"Step {time_step}   "
            f"L2: {fm['each_step_L2'][time_step]:.4e}    "
            f"RMSE: {fm['each_step_RMSE'][time_step]:.4e}    "
            f"IoU: {fm['IoU_per_step'][time_step]:.4f}    "
            f"Dice: {fm['Dice_per_step'][time_step]:.4f}    "
            f"Band MAE: {fm['band_MAE_per_step'][time_step]:.4e}"
        )
        line_mean = (
            f"Overall   "
            f"MSE(norm): {metrics['MSE_normalized']:.4e}    "
            f"Rel L2: {fm['relative_L2']:.4e}    "
            f"RMSE: {fm['RMSE']:.4e}    "
            f"Mean IoU: {fm['mean_IoU']:.4f}    "
            f"Mean Dice: {fm['mean_Dice']:.4f}    "
            f"Mean Band MAE: {fm['mean_band_MAE']:.4e}"
        )
        return line_step, line_mean

    def _contour_outlined(self, ax, X, Y, Z, levels, color, lw, ls="-", zorder=5):
        ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=lw + 1.5, linestyles=ls, zorder=zorder)
        return ax.contour(
            X, Y, Z, levels=levels, colors=color, linewidths=lw, linestyles=ls, zorder=zorder + 1
        )

    def plot_generic_field(
        self,
        result_dict,
        metrics,
        time_step,
        field_name,
        axis="z",
        slice_pos=None,
        res=320,
        save_path=None,
        return_array=False,
        smooth_sigma=0.9,
        rel_err_cap=1.0,
    ):
        field_idx = self.fields.index(field_name)
        coords = np.asarray(result_dict["coords"])
        pred_data = result_dict["pred"][time_step, :, field_idx]
        gt_data = result_dict["gt"][time_step, :, field_idx]

        gt_int = None if result_dict.get("gt_interface") is None else result_dict["gt_interface"][time_step]
        pred_int = None if result_dict.get("pred_interface") is None else result_dict["pred_interface"][time_step]

        mask, _, pts_x, pts_y, xlabel, ylabel = self._project_slice_points(
            coords, axis=axis, slice_pos=slice_pos, min_points=32
        )
        if np.sum(mask) < 4 or len(np.unique(pts_x)) < 2 or len(np.unique(pts_y)) < 2:
            print(f"[Warn] Too few points for field '{field_name}', sample slice skipped.")
            return None

        xi = np.linspace(pts_x.min(), pts_x.max(), res)
        yi = np.linspace(pts_y.min(), pts_y.max(), res)
        Xi, Yi = np.meshgrid(xi, yi)

        is_phase = _is_vof_field(field_name)
        interp_method = "linear" if is_phase else "cubic"
        sigma = 0.5 if is_phase else smooth_sigma

        Zi_pred_raw = self._interp_grid(pts_x, pts_y, pred_data[mask], Xi, Yi, method=interp_method)
        Zi_gt_raw = self._interp_grid(pts_x, pts_y, gt_data[mask], Xi, Yi, method=interp_method)
        if Zi_pred_raw is None or Zi_gt_raw is None:
            print(f"[Warn] Interpolation failed for field '{field_name}', time step {time_step}.")
            return None

        Zi_pred = self._smooth(Zi_pred_raw, sigma=sigma)
        Zi_gt = self._smooth(Zi_gt_raw, sigma=sigma)
        if is_phase:
            Zi_pred = np.clip(Zi_pred, 0, 1)
            Zi_gt = np.clip(Zi_gt, 0, 1)

        Zi_err = np.abs(Zi_pred - Zi_gt)
        denom_floor = max(1e-8, 1e-3 * np.nanmax(np.abs(Zi_gt)))
        Zi_rel_err = Zi_err / np.maximum(np.abs(Zi_gt), denom_floor)
        Zi_rel_err = np.clip(Zi_rel_err, 0, rel_err_cap)

        Zi_err_vis = self._smooth(Zi_err, sigma=min(max(sigma, 0.3), 1.0))
        Zi_rel_err_vis = self._smooth(Zi_rel_err, sigma=min(max(sigma, 0.3), 1.0))

        if gt_int is not None:
            Zi_gt_int = self._interp_grid(pts_x, pts_y, gt_int[mask], Xi, Yi, method="linear")
        else:
            Zi_gt_int = None
        if pred_int is not None:
            Zi_pred_int = self._interp_grid(pts_x, pts_y, pred_int[mask], Xi, Yi, method="linear")
        else:
            Zi_pred_int = None

        vmin, vmax = self._field_limits(result_dict, field_name, field_idx)
        cmap = "inferno" if "T" in field_name else "viridis"
        imshow_args = dict(
            extent=(xi.min(), xi.max(), yi.min(), yi.max()),
            origin="lower",
            interpolation="bicubic",
            aspect="equal",
        )

        text_step, text_mean = self._generic_metrics_text(metrics, field_name, time_step)

        plt.rcParams.update({
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        })

        fig = plt.figure(figsize=(22, 8.2))
        gs = fig.add_gridspec(2, 4, height_ratios=[6, 1.15], hspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        im0 = axes[0].imshow(Zi_gt, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_args)
        axes[0].set_title(f"Ground Truth ({field_name})")
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
        cb0.set_label(field_name)

        im1 = axes[1].imshow(Zi_pred, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_args)
        axes[1].set_title(f"Prediction ({field_name})")
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
        cb1.set_label(field_name)

        err_vmax = np.nanpercentile(Zi_err_vis, 99.0) if np.any(np.isfinite(Zi_err_vis)) else 1.0
        err_vmax = max(err_vmax, 1e-12)
        im2 = axes[2].imshow(Zi_err_vis, cmap="magma", vmin=0, vmax=err_vmax, **imshow_args)
        axes[2].set_title("Absolute Error")
        cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03, extend="max")
        cb2.set_label("|Pred - GT|")

        rel_vmax = min(rel_err_cap, max(0.1, np.nanpercentile(Zi_rel_err_vis, 99.0)))
        im3 = axes[3].imshow(Zi_rel_err_vis, cmap="magma", vmin=0, vmax=rel_vmax, **imshow_args)
        axes[3].set_title(f"Relative Error (clip={rel_err_cap:.2f})")
        cb3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.03, extend="max")
        cb3.set_label("|Pred-GT| / max(|GT|, eps)")

        legend_lines = []
        legend_labels = []
        if Zi_gt_int is not None:
            axes[0].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors="white", linestyles="--", linewidths=1.5)
            axes[1].contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors="white", linestyles="--", linewidths=1.5)
            legend_lines.append(Line2D([0], [0], color="white", linestyle="--", linewidth=1.5))
            legend_labels.append("GT Interface")

        if Zi_pred_int is not None:
            axes[1].contour(Xi, Yi, Zi_pred_int, levels=[0.5], colors="red", linestyles=":", linewidths=1.5)
            legend_lines.append(Line2D([0], [0], color="red", linestyle=":", linewidth=1.5))
            legend_labels.append("Pred Interface")

        for ax in (axes[2], axes[3]):
            if Zi_gt_int is not None:
                ax.contour(Xi, Yi, Zi_gt_int, levels=[0.5], colors="white", linestyles="--", linewidths=1.5)
            if Zi_pred_int is not None:
                ax.contour(Xi, Yi, Zi_pred_int, levels=[0.5], colors="red", linestyles=":", linewidths=1.5)
            if legend_lines:
                ax.legend(legend_lines, legend_labels, loc="upper right", framealpha=0.7, fontsize="small")

        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="both")
            ax.set_aspect("equal", adjustable="box")

        fig.suptitle(f"{field_name} -- Step {time_step}", fontsize=14, fontweight="bold")

        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis("off")
        ax_text.text(
            0.01,
            0.95,
            text_step,
            transform=ax_text.transAxes,
            fontsize=11.5,
            fontfamily="monospace",
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe", edgecolor="#4a90d9", alpha=0.92),
        )
        ax_text.text(
            0.01,
            0.38,
            text_mean,
            transform=ax_text.transAxes,
            fontsize=10.8,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.90),
        )

        if return_array:
            return self._render_array(fig)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[Saved] {save_path}")
            return None
        plt.show()
        plt.close(fig)
        return None

    def plot_vof_field(
        self,
        result_dict,
        metrics,
        time_step,
        field_name,
        axis="z",
        slice_pos=None,
        res=320,
        save_path=None,
        return_array=False,
        smooth_sigma=0.5,
    ):
        field_idx = self.fields.index(field_name)
        coords = np.asarray(result_dict["coords"])
        pred_data = result_dict["pred"][time_step, :, field_idx]
        gt_data = result_dict["gt"][time_step, :, field_idx]

        mask, _, pts_x, pts_y, xlabel, ylabel = self._project_slice_points(
            coords, axis=axis, slice_pos=slice_pos, min_points=32
        )
        if np.sum(mask) < 4 or len(np.unique(pts_x)) < 2 or len(np.unique(pts_y)) < 2:
            print(f"[Warn] Too few points for VOF field '{field_name}', sample slice skipped.")
            return None

        xi = np.linspace(pts_x.min(), pts_x.max(), res)
        yi = np.linspace(pts_y.min(), pts_y.max(), res)
        Xi, Yi = np.meshgrid(xi, yi)

        Zi_gt_raw = self._interp_grid(pts_x, pts_y, gt_data[mask], Xi, Yi, method="linear")
        Zi_pred_raw = self._interp_grid(pts_x, pts_y, pred_data[mask], Xi, Yi, method="linear")
        if Zi_gt_raw is None or Zi_pred_raw is None:
            print(f"[Warn] Interpolation failed for VOF field '{field_name}', time step {time_step}.")
            return None

        Zi_gt = np.clip(self._smooth(Zi_gt_raw, sigma=smooth_sigma), 0, 1)
        Zi_pred = np.clip(self._smooth(Zi_pred_raw, sigma=smooth_sigma), 0, 1)

        extent = (xi.min(), xi.max(), yi.min(), yi.max())
        imshow_args = dict(extent=extent, origin="lower", aspect="equal")
        text_step, text_mean = self._vof_metrics_text(metrics, field_name, time_step)

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.facecolor": "#f8f8f8",
            "figure.facecolor": "white",
        })

        fig = plt.figure(figsize=(26, 8.5))
        gs = fig.add_gridspec(2, 4, height_ratios=[6, 1.2], hspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        diff_field = Zi_pred - Zi_gt
        im0 = axes[0].imshow(diff_field, cmap="RdBu_r", vmin=-0.1, vmax=0.1, interpolation="bicubic", **imshow_args)

        gt_above = Zi_gt >= 0.5
        pred_above = Zi_pred >= 0.5
        disagree = (gt_above != pred_above).astype(float)
        disagree_masked = np.where(disagree > 0.5, 1.0, np.nan)
        axes[0].imshow(
            disagree_masked, cmap="Oranges", vmin=0, vmax=2, alpha=0.45, interpolation="nearest", **imshow_args
        )

        self._contour_outlined(axes[0], Xi, Yi, Zi_gt, [0.5], color="#FFD700", lw=1.8, ls="-")
        self._contour_outlined(axes[0], Xi, Yi, Zi_pred, [0.5], color="#FF00FF", lw=1.8, ls="--")
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, extend="both")
        cb0.set_label("Pred - GT")
        axes[0].set_title("Interface Comparison", fontweight="bold")
        axes[0].legend(
            handles=[
                Line2D([0], [0], color="#FFD700", lw=2, ls="-", label="GT (0.5)"),
                Line2D([0], [0], color="#FF00FF", lw=2, ls="--", label="Pred (0.5)"),
                Line2D([0], [0], color="#e8871e", lw=6, ls="-", alpha=0.45, label="Mismatch"),
            ],
            loc="upper right",
            framealpha=0.85,
            fontsize=9,
        )

        im1 = axes[1].imshow(Zi_gt, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="bicubic", **imshow_args)
        self._contour_outlined(axes[1], Xi, Yi, Zi_gt, [0.5], color="white", lw=1.8, ls="-")
        axes[1].set_title(f"GT {field_name}")
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
        cb1.set_label(field_name)

        im2 = axes[2].imshow(Zi_pred, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="bicubic", **imshow_args)
        self._contour_outlined(axes[2], Xi, Yi, Zi_pred, [0.5], color="white", lw=1.8, ls="-")
        self._contour_outlined(axes[2], Xi, Yi, Zi_gt, [0.5], color="#FFD700", lw=1.5, ls="--")
        axes[2].set_title(f"Pred {field_name}")
        cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
        cb2.set_label(field_name)
        axes[2].legend(
            handles=[
                Line2D([0], [0], color="white", lw=1.8, ls="-", label="Pred 0.5"),
                Line2D([0], [0], color="#FFD700", lw=1.5, ls="--", label="GT 0.5"),
            ],
            loc="upper right",
            framealpha=0.85,
            fontsize=9,
        )

        abs_err = np.abs(Zi_pred - Zi_gt)
        interface_region = (Zi_gt > 0.1) & (Zi_gt < 0.9)
        err_masked = np.where(interface_region, abs_err, np.nan)
        err_vmax = np.nanpercentile(err_masked, 99) if np.any(interface_region) else 0.1
        err_vmax = max(err_vmax, 0.01)
        im3 = axes[3].imshow(err_masked, cmap="inferno", vmin=0, vmax=err_vmax, interpolation="bicubic", **imshow_args)
        self._contour_outlined(axes[3], Xi, Yi, Zi_gt, [0.5], color="cyan", lw=1.8, ls="-")
        self._contour_outlined(axes[3], Xi, Yi, Zi_pred, [0.5], color="lime", lw=1.8, ls="--")
        if np.any(disagree > 0.5):
            axes[3].contour(Xi, Yi, disagree, levels=[0.5], colors="yellow", linewidths=1.0, linestyles=":", alpha=0.7)
        axes[3].set_title("Interface Error (|Pred-GT|)")
        cb3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.03, extend="max")
        cb3.set_label(f"|Pred - GT| ({field_name})")
        axes[3].legend(
            handles=[
                Line2D([0], [0], color="cyan", lw=1.8, ls="-", label="GT 0.5"),
                Line2D([0], [0], color="lime", lw=1.8, ls="--", label="Pred 0.5"),
            ],
            loc="upper right",
            framealpha=0.85,
            fontsize=9,
        )

        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="both")
            ax.set_aspect("equal", adjustable="box")

        fig.suptitle(f"{field_name} -- Step {time_step}", fontsize=14, fontweight="bold")

        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis("off")
        ax_text.text(
            0.01,
            0.95,
            text_step,
            transform=ax_text.transAxes,
            fontsize=12,
            fontfamily="monospace",
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe", edgecolor="#4a90d9", alpha=0.92),
        )
        ax_text.text(
            0.01,
            0.38,
            text_mean,
            transform=ax_text.transAxes,
            fontsize=11,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.90),
        )

        if return_array:
            return self._render_array(fig)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[Saved] {save_path}")
            return None
        plt.show()
        plt.close(fig)
        return None

    def plot_field(
        self,
        result_dict,
        metrics,
        time_step,
        field_name,
        axis="z",
        slice_pos=None,
        res=320,
        save_path=None,
        return_array=False,
    ):
        if _is_vof_field(field_name):
            return self.plot_vof_field(
                result_dict=result_dict,
                metrics=metrics,
                time_step=time_step,
                field_name=field_name,
                axis=axis,
                slice_pos=slice_pos,
                res=res,
                save_path=save_path,
                return_array=return_array,
            )
        return self.plot_generic_field(
            result_dict=result_dict,
            metrics=metrics,
            time_step=time_step,
            field_name=field_name,
            axis=axis,
            slice_pos=slice_pos,
            res=res,
            save_path=save_path,
            return_array=return_array,
        )

    def generate_gif(
        self,
        result_dict,
        metrics,
        field_name,
        axis="z",
        slice_pos=None,
        gif_path="rollout.gif",
        res=280,
        fps=8,
    ):
        horizon = result_dict["pred"].shape[0]
        frames = []
        print(f"[GIF] Generating {field_name}: {gif_path}")
        for t in tqdm(range(horizon), desc=f"Rendering {field_name}", leave=False):
            frame = self.plot_field(
                result_dict=result_dict,
                metrics=metrics,
                time_step=t,
                field_name=field_name,
                axis=axis,
                slice_pos=slice_pos,
                res=res,
                return_array=True,
            )
            if frame is not None:
                frames.append(frame)

        if not frames:
            print(f"[GIF] No frames generated for {field_name}.")
            return

        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"[GIF] Saved to {gif_path}")

    def format_sample_metrics(self, metrics):
        lines = []
        lines.append("=" * 80)
        lines.append(f"Sample {metrics['sample_idx']}")
        lines.append("=" * 80)
        lines.append(f"MSE_normalized : {_fmt_scalar(metrics['MSE_normalized'])}")
        lines.append(f"mean_L2_all    : {_fmt_scalar(metrics['mean_L2_all'])}")
        lines.append(f"each_step_L2_all   : {_fmt_array(metrics['each_step_L2_all'])}")
        lines.append(f"each_step_RMSE_all : {_fmt_array(metrics['each_step_RMSE_all'])}")

        if "active_mean_L2" in metrics:
            lines.append(f"active_mean_L2   : {_fmt_scalar(metrics['active_mean_L2'])}")
            lines.append(f"inactive_mean_L2 : {_fmt_scalar(metrics['inactive_mean_L2'])}")

        for field_name in self.fields:
            fm = metrics["per_field"][field_name]
            lines.append("")
            lines.append(f"[Field] {field_name} ({fm['field_type']})")
            lines.append(f"relative_L2      : {_fmt_scalar(fm['relative_L2'])}")
            lines.append(f"RMSE             : {_fmt_scalar(fm['RMSE'])}")
            lines.append(f"each_step_L2     : {_fmt_array(fm['each_step_L2'])}")
            lines.append(f"each_step_RMSE   : {_fmt_array(fm['each_step_RMSE'])}")

            if "active_L2" in fm:
                lines.append(f"active_L2        : {_fmt_scalar(fm['active_L2'])}")
                lines.append(f"inactive_L2      : {_fmt_scalar(fm['inactive_L2'])}")
                lines.append(f"active_RMSE      : {_fmt_scalar(fm['active_RMSE'])}")
                lines.append(f"inactive_RMSE    : {_fmt_scalar(fm['inactive_RMSE'])}")

            if fm["field_type"] == "vof":
                lines.append(f"mean_IoU         : {_fmt_scalar(fm['mean_IoU'], sci=False)}")
                lines.append(f"mean_Dice        : {_fmt_scalar(fm['mean_Dice'], sci=False)}")
                lines.append(f"mean_band_MAE    : {_fmt_scalar(fm['mean_band_MAE'])}")
                lines.append(f"IoU_per_step     : {_fmt_array(fm['IoU_per_step'], sci=False)}")
                lines.append(f"Dice_per_step    : {_fmt_array(fm['Dice_per_step'], sci=False)}")
                lines.append(f"band_MAE_per_step: {_fmt_array(fm['band_MAE_per_step'])}")

        return "\n".join(lines) + "\n"

    def build_summary_text(self, all_metrics, cfg_path, sample_idxs, failed_samples=None):
        failed_samples = failed_samples or []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append("#" * 100)
        lines.append("Inference Summary")
        lines.append("#" * 100)
        lines.append(f"Time        : {now_str}")
        lines.append(f"Config      : {cfg_path}")
        lines.append(f"Run Name    : {self.args.name}")
        lines.append(f"Mode        : {self.mode}")
        lines.append(f"Checkpoint  : {self.model_path}")
        lines.append(f"Fields      : {self.fields}")
        lines.append(f"Sample Idxs : {sample_idxs}")
        if failed_samples:
            lines.append(f"Failed      : {failed_samples}")
        lines.append("")

        for metrics in all_metrics:
            lines.append(self.format_sample_metrics(metrics).rstrip())
            lines.append("")

        if not all_metrics:
            lines.append("No successful samples.")
            return "\n".join(lines) + "\n"

        mse_vals = np.asarray([m["MSE_normalized"] for m in all_metrics], dtype=np.float64)
        mean_l2_vals = np.asarray([m["mean_L2_all"] for m in all_metrics], dtype=np.float64)
        each_l2_all = np.stack([m["each_step_L2_all"] for m in all_metrics], axis=0)
        each_rmse_all = np.stack([m["each_step_RMSE_all"] for m in all_metrics], axis=0)

        lines.append("#" * 100)
        lines.append(f"Aggregate over {len(all_metrics)} samples")
        lines.append("#" * 100)
        lines.append(
            f"MSE_normalized : mean={np.mean(mse_vals):.4e}  std={np.std(mse_vals):.4e}  "
            f"min={np.min(mse_vals):.4e}  max={np.max(mse_vals):.4e}"
        )
        lines.append(
            f"mean_L2_all    : mean={np.mean(mean_l2_vals):.4e}  std={np.std(mean_l2_vals):.4e}  "
            f"min={np.min(mean_l2_vals):.4e}  max={np.max(mean_l2_vals):.4e}"
        )
        lines.append(f"each_step_L2_all_mean   : {_fmt_array(np.mean(each_l2_all, axis=0))}")
        lines.append(f"each_step_RMSE_all_mean : {_fmt_array(np.mean(each_rmse_all, axis=0))}")

        if "active_mean_L2" in all_metrics[0]:
            active_vals = np.asarray([m["active_mean_L2"] for m in all_metrics], dtype=np.float64)
            inactive_vals = np.asarray([m["inactive_mean_L2"] for m in all_metrics], dtype=np.float64)
            lines.append(
                f"active_mean_L2   : mean={np.nanmean(active_vals):.4e}  std={np.nanstd(active_vals):.4e}"
            )
            lines.append(
                f"inactive_mean_L2 : mean={np.nanmean(inactive_vals):.4e}  std={np.nanstd(inactive_vals):.4e}"
            )

        for field_name in self.fields:
            field_metrics = [m["per_field"][field_name] for m in all_metrics]
            rel_vals = np.asarray([fm["relative_L2"] for fm in field_metrics], dtype=np.float64)
            rmse_vals = np.asarray([fm["RMSE"] for fm in field_metrics], dtype=np.float64)
            each_l2 = np.stack([fm["each_step_L2"] for fm in field_metrics], axis=0)
            each_rmse = np.stack([fm["each_step_RMSE"] for fm in field_metrics], axis=0)

            lines.append("")
            lines.append(f"[Aggregate Field] {field_name}")
            lines.append(
                f"relative_L2 : mean={np.mean(rel_vals):.4e}  std={np.std(rel_vals):.4e}  "
                f"min={np.min(rel_vals):.4e}  max={np.max(rel_vals):.4e}"
            )
            lines.append(
                f"RMSE        : mean={np.mean(rmse_vals):.4e}  std={np.std(rmse_vals):.4e}  "
                f"min={np.min(rmse_vals):.4e}  max={np.max(rmse_vals):.4e}"
            )
            lines.append(f"each_step_L2_mean   : {_fmt_array(np.mean(each_l2, axis=0))}")
            lines.append(f"each_step_RMSE_mean : {_fmt_array(np.mean(each_rmse, axis=0))}")

            if "active_L2" in field_metrics[0]:
                active_l2 = np.asarray([fm["active_L2"] for fm in field_metrics], dtype=np.float64)
                inactive_l2 = np.asarray([fm["inactive_L2"] for fm in field_metrics], dtype=np.float64)
                active_rmse = np.asarray([fm["active_RMSE"] for fm in field_metrics], dtype=np.float64)
                inactive_rmse = np.asarray([fm["inactive_RMSE"] for fm in field_metrics], dtype=np.float64)
                lines.append(
                    f"active_L2   : mean={np.nanmean(active_l2):.4e}  std={np.nanstd(active_l2):.4e}"
                )
                lines.append(
                    f"inactive_L2 : mean={np.nanmean(inactive_l2):.4e}  std={np.nanstd(inactive_l2):.4e}"
                )
                lines.append(
                    f"active_RMSE : mean={np.nanmean(active_rmse):.4e}  std={np.nanstd(active_rmse):.4e}"
                )
                lines.append(
                    f"inactive_RMSE : mean={np.nanmean(inactive_rmse):.4e}  std={np.nanstd(inactive_rmse):.4e}"
                )

            if field_metrics[0]["field_type"] == "vof":
                mean_iou = np.asarray([fm["mean_IoU"] for fm in field_metrics], dtype=np.float64)
                mean_dice = np.asarray([fm["mean_Dice"] for fm in field_metrics], dtype=np.float64)
                mean_band = np.asarray([fm["mean_band_MAE"] for fm in field_metrics], dtype=np.float64)
                iou_steps = np.stack([fm["IoU_per_step"] for fm in field_metrics], axis=0)
                dice_steps = np.stack([fm["Dice_per_step"] for fm in field_metrics], axis=0)
                band_steps = np.stack([fm["band_MAE_per_step"] for fm in field_metrics], axis=0)
                lines.append(
                    f"mean_IoU      : mean={np.mean(mean_iou):.4f}  std={np.std(mean_iou):.4f}"
                )
                lines.append(
                    f"mean_Dice     : mean={np.mean(mean_dice):.4f}  std={np.std(mean_dice):.4f}"
                )
                lines.append(
                    f"mean_band_MAE : mean={np.mean(mean_band):.4e}  std={np.std(mean_band):.4e}"
                )
                lines.append(f"IoU_per_step_mean      : {_fmt_array(np.mean(iou_steps, axis=0), sci=False)}")
                lines.append(f"Dice_per_step_mean     : {_fmt_array(np.mean(dice_steps, axis=0), sci=False)}")
                lines.append(f"band_MAE_per_step_mean : {_fmt_array(np.mean(band_steps, axis=0))}")

        return "\n".join(lines) + "\n"


def _resolve_config_paths(parsed_args):
    configs = []
    if parsed_args.config:
        configs.extend(parsed_args.config)
    if parsed_args.configs:
        configs.extend(parsed_args.configs)
    configs = [str(Path(cfg).expanduser()) for cfg in configs]
    if not configs:
        raise ValueError("请通过位置参数或 --config 传入至少一个配置文件。")
    return configs


def _resolve_sample_indices(dataset_length, explicit_sample_idxs, num_samples):
    if explicit_sample_idxs:
        valid = []
        for idx in explicit_sample_idxs:
            if 0 <= idx < dataset_length:
                valid.append(idx)
            else:
                print(f"[Warn] sample_idx={idx} 超出范围 [0, {dataset_length - 1}]，已跳过。")
        return valid
    count = min(max(1, num_samples), dataset_length)
    return random.sample(range(dataset_length), count)


def _resolve_interface_field(cli_value, predictor: InferenceMaxPredictor):
    if cli_value is not None:
        lowered = cli_value.lower()
        if lowered in ("none", "null", "no", "off"):
            return None
        return cli_value
    if predictor.vof_fields:
        return predictor.vof_fields[0][0]
    return "alpha.air"


def _resolve_fields_to_render(cli_fields, predictor: InferenceMaxPredictor):
    if not cli_fields:
        return list(predictor.fields)
    fields = []
    for field in cli_fields:
        if field in predictor.fields:
            fields.append(field)
        else:
            print(f"[Warn] Field '{field}' 不在配置字段中，已跳过。")
    return fields


DEFAULT_INFERENCE_CONFIG = {
    "config_paths": [
        "config/new_liquid/easypool/GTO_a3_ep_s_liquid.json",
        "config/new_liquid/easypool/GTO_a3_ep_s_T_liquid.json",
        "config/new_liquid/easypool/GTO_ep_s_T_liquid.json",
        "config/new_liquid/keyhole/GTO_a3_kh_s_liquid.json",
        "config/new_liquid/keyhole/GTO_a3_kh_s_T_liquid.json"
    ],
    "mode": "test",
    "output_dir_name": "inference",
    "sample_idxs": None,
    "num_samples": 3,
    "field_names": None,
    "interface_field": None,
    "slice_axis": "z",
    "slice_pos": 5e-4,
    "res": 280,
    "gif_fps": 4,
    "device": "cuda:0",
    "model_path": None,
    "seed": 42,
    "skip_gif": False,
}


def parse_cli():
    parser = argparse.ArgumentParser(description="Unified inference script for PhysGTO models.")
    parser.add_argument("configs", nargs="*", help="配置文件路径，可一次传多个。")
    parser.add_argument("--config", nargs="+", help="配置文件路径，可一次传多个。")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir-name", default="inference", help="保存到 save_path 下的目录名。")
    parser.add_argument("--sample-idxs", nargs="*", type=int, default=None, help="指定多个样本索引。")
    parser.add_argument("--num-samples", type=int, default=3, help="未指定 sample_idxs 时随机抽样数量。")
    parser.add_argument("--field-names", nargs="*", default=None, help="仅渲染这些字段；默认渲染全部字段。")
    parser.add_argument("--interface-field", default=None, help="非体积分数字段叠加的界面字段，默认自动选择。")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--slice-pos", type=float, default=None)
    parser.add_argument("--res", type=int, default=280, help="插值分辨率。")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--device", default=None, help="覆盖配置中的 device。")
    parser.add_argument("--model-path", default=None, help="手动指定权重路径。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-gif", action="store_true", help="只计算指标，不生成 GIF。")
    return parser.parse_args()


def _normalize_runtime_options(runtime_options):
    options = dict(DEFAULT_INFERENCE_CONFIG)
    if runtime_options:
        options.update(runtime_options)

    config_paths = options.get("config_paths", [])
    if isinstance(config_paths, (str, Path)):
        config_paths = [str(config_paths)]
    else:
        config_paths = [str(Path(cfg).expanduser()) for cfg in config_paths]
    options["config_paths"] = config_paths

    sample_idxs = options.get("sample_idxs")
    if sample_idxs is not None:
        options["sample_idxs"] = list(sample_idxs)

    field_names = options.get("field_names")
    if field_names is not None:
        options["field_names"] = list(field_names)

    return options


def _runtime_options_from_cli(cli_args):
    config_paths = []
    if cli_args.config:
        config_paths.extend(cli_args.config)
    if cli_args.configs:
        config_paths.extend(cli_args.configs)

    return _normalize_runtime_options({
        "config_paths": config_paths,
        "mode": cli_args.mode,
        "output_dir_name": cli_args.output_dir_name,
        "sample_idxs": cli_args.sample_idxs,
        "num_samples": cli_args.num_samples,
        "field_names": cli_args.field_names,
        "interface_field": cli_args.interface_field,
        "slice_axis": cli_args.slice_axis,
        "slice_pos": cli_args.slice_pos,
        "res": cli_args.res,
        "gif_fps": cli_args.gif_fps,
        "device": cli_args.device,
        "model_path": cli_args.model_path,
        "seed": cli_args.seed,
        "skip_gif": cli_args.skip_gif,
    })


def run_for_config(cfg_path, runtime_options):
    predictor = InferenceMaxPredictor(
        config_path=cfg_path,
        mode=runtime_options["mode"],
        model_path=runtime_options["model_path"],
        device_str=runtime_options["device"],
    )

    cfg_name = Path(cfg_path).stem
    output_root = Path(predictor.args.save_path) / runtime_options["output_dir_name"] / cfg_name
    output_root.mkdir(parents=True, exist_ok=True)
    gif_dir = output_root / "gif"
    txt_dir = output_root / "txt"
    gif_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    dataset_length = len(predictor.dataset)
    sample_idxs = _resolve_sample_indices(dataset_length, runtime_options["sample_idxs"], runtime_options["num_samples"])
    if not sample_idxs:
        raise ValueError("没有可用的 sample_idx 可推理。")

    interface_field = _resolve_interface_field(runtime_options["interface_field"], predictor)
    render_fields = _resolve_fields_to_render(runtime_options["field_names"], predictor)
    if not render_fields and not runtime_options["skip_gif"]:
        raise ValueError("没有可渲染的字段。")

    print(f"[Run] save_path root : {predictor.args.save_path}")
    print(f"[Run] output dir     : {output_root}")
    print(f"[Run] gif dir        : {gif_dir}")
    print(f"[Run] txt dir        : {txt_dir}")
    print(f"[Run] sample idxs    : {sample_idxs}")
    print(f"[Run] interface field: {interface_field}")
    print(f"[Run] render fields  : {render_fields}")

    all_metrics = []
    failed_samples = []

    for sample_idx in sample_idxs:
        print(f"\n>>> Inference on sample {sample_idx}")

        try:
            result, metrics = predictor.predict_and_evaluate(sample_idx, interface_field=interface_field)
            all_metrics.append(metrics)

            sample_txt = txt_dir / f"metrics_sample_{sample_idx:06d}.txt"
            # sample_txt.write_text(predictor.format_sample_metrics(metrics), encoding="utf-8")
            # print(f"[Saved] {sample_txt}")

            if not runtime_options["skip_gif"]:
                for field_name in render_fields:
                    gif_path = gif_dir / f"rollout_sample{sample_idx}_{_safe_name(field_name)}.gif"
                    predictor.generate_gif(
                        result_dict=result,
                        metrics=metrics,
                        field_name=field_name,
                        axis=runtime_options["slice_axis"],
                        slice_pos=runtime_options["slice_pos"],
                        gif_path=str(gif_path),
                        res=runtime_options["res"],
                        fps=runtime_options["gif_fps"],
                    )
        except Exception as exc:
            failed_samples.append(sample_idx)
            print(f"[Error] sample {sample_idx} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = txt_dir / f"metrics_summary_{timestamp}.txt"
    summary_text = predictor.build_summary_text(
        all_metrics=all_metrics,
        cfg_path=str(Path(cfg_path).expanduser().resolve()),
        sample_idxs=sample_idxs,
        failed_samples=failed_samples,
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"[Saved] {summary_path}")


def run_from_config(runtime_options=None):
    runtime_options = _normalize_runtime_options(runtime_options)
    cfg_paths = runtime_options["config_paths"]
    if not cfg_paths:
        raise ValueError("请先在 DEFAULT_INFERENCE_CONFIG['config_paths'] 中填写配置文件路径。")

    for cfg_path in cfg_paths:
        try:
            run_for_config(cfg_path, runtime_options)
        except Exception as exc:
            print(f"[Fatal] config {cfg_path} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()


def main():
    runtime_options = _normalize_runtime_options(DEFAULT_INFERENCE_CONFIG)
    set_seed(runtime_options["seed"])
    run_from_config(runtime_options)


def main_cli():
    cli_args = parse_cli()
    runtime_options = _runtime_options_from_cli(cli_args)
    set_seed(runtime_options["seed"])
    if not runtime_options["config_paths"]:
        runtime_options["config_paths"] = list(DEFAULT_INFERENCE_CONFIG["config_paths"])
    run_from_config(runtime_options)


if __name__ == "__main__":
    main()
