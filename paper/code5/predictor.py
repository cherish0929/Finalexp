# -*- coding: utf-8 -*-
"""
predictor.py — paper/code5 公用推理工具

围绕 inference_v1.AeroGtoPredictor 做轻量封装，供 4.5 节三类分析脚本统一调用：
    1) 4.5.1 误差空间分布热力图 (error_spatial.py)
    2) 4.5.2 工况敏感性分析 (sensitivity.py)
    3) 4.5.3 失败案例剖析 (failure_case.py)

公用接口：
    - load_predictor(config_path)           : 复用 inference_v1 的 Predictor
    - rollout_sample(predictor, idx)        : 单样本自回归推理，返回 pred/gt/coords/conds
    - per_sample_metrics(predictor, idx)    : 全场 / 活跃区 / 非活跃区 rL2 与逐通道 RMSE
    - get_condition_params(h5_path)         : 从 h5 读取 (P, r0, A, v) 并计算 E_v
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch

# 让脚本在 paper/code5 下也能 import 到根目录的模块
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference_v1 import AeroGtoPredictor  # noqa: E402
from src.utils import build_active_mask  # noqa: E402


# ----------------------------------------------------------------------------- #
# Predictor 构造 / 复用
# ----------------------------------------------------------------------------- #
def load_predictor(config_path: str,
                   mode: str = "test",
                   model_path: Optional[str] = None,
                   device_str: str = "cuda") -> AeroGtoPredictor:
    """构建 Predictor。封装一次以便后续脚本复用。"""
    return AeroGtoPredictor(config_path, mode=mode, model_path=model_path,
                            device_str=device_str)


# ----------------------------------------------------------------------------- #
# 单样本 rollout（复用 inference_v1.predict_rollout）
# ----------------------------------------------------------------------------- #
def rollout_sample(predictor: AeroGtoPredictor,
                   idx: int,
                   interface_field: str = "alpha.air") -> Dict:
    """对索引 idx 的样本执行自回归推理。返回字典包含真实物理量级的 pred/gt 与坐标。"""
    return predictor.predict_rollout(sample_idx=idx, interface_field=interface_field)


# ----------------------------------------------------------------------------- #
# 单样本指标：全场 / 活跃区 / 非活跃区 rL2，逐通道 rL2 与 RMSE
# ----------------------------------------------------------------------------- #
def _safe_rl2(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    num = np.linalg.norm(pred - gt)
    den = np.linalg.norm(gt) + eps
    return float(num / den)


def per_sample_metrics(predictor: AeroGtoPredictor,
                       idx: int,
                       interface_field: str = "alpha.air") -> Dict:
    """
    计算单样本的精度指标，返回结构：
        {
            'idx', 'fields',
            'rl2_field': {field: float, ...},
            'rmse_field': {field: float, ...},
            'rl2_overall': float,
            'rl2_active': float,
            'rl2_inactive': float,
            'pred_real', 'gt_real', 'coords', 'active_mask',
        }
    其中 active_mask 由 src.utils.build_active_mask 在归一化空间得到（与训练一致）。
    """
    out = predictor.predict_rollout(sample_idx=idx, interface_field=interface_field)
    pred_real = out["pred"]   # [T, N, C]
    gt_real = out["gt"]       # [T, N, C]
    coords = out["coords"]    # [N, 3]
    fields = list(predictor.fields)

    # 活跃区掩码：在归一化数据上判定，与训练时保持一致。
    norm_mean = predictor.normalizer.mean.cpu().numpy().reshape(1, 1, -1)
    norm_std = (predictor.normalizer.std + predictor.normalizer.eps).cpu().numpy().reshape(1, 1, -1)
    gt_norm = (gt_real - norm_mean) / norm_std  # [T, N, C]
    mask_cfg = predictor.dataset.mask_cfg
    active_mask = build_active_mask(gt_norm, fields, mask_cfg)  # [T, N] or [T, N, 1]
    if active_mask is None:
        active_mask = np.ones(gt_real.shape[:2], dtype=bool)
    else:
        if torch.is_tensor(active_mask):
            active_mask = active_mask.cpu().numpy()
        active_mask = active_mask.astype(bool).reshape(gt_real.shape[:2])

    rl2_field, rmse_field = {}, {}
    for c, fname in enumerate(fields):
        rl2_field[fname] = _safe_rl2(pred_real[..., c], gt_real[..., c])
        rmse_field[fname] = float(np.sqrt(np.mean((pred_real[..., c] - gt_real[..., c]) ** 2)))

    rl2_overall = float(np.mean([rl2_field[f] for f in fields]))

    if active_mask.any():
        m3 = np.repeat(active_mask[..., None], len(fields), axis=-1)
        rl2_active = _safe_rl2(pred_real[m3], gt_real[m3])
    else:
        rl2_active = float("nan")

    inactive_mask = ~active_mask
    if inactive_mask.any():
        m3 = np.repeat(inactive_mask[..., None], len(fields), axis=-1)
        rl2_inactive = _safe_rl2(pred_real[m3], gt_real[m3])
    else:
        rl2_inactive = float("nan")

    return {
        "idx": int(idx),
        "fields": fields,
        "rl2_field": rl2_field,
        "rmse_field": rmse_field,
        "rl2_overall": rl2_overall,
        "rl2_active": rl2_active,
        "rl2_inactive": rl2_inactive,
        "pred_real": pred_real,
        "gt_real": gt_real,
        "coords": coords,
        "active_mask": active_mask,
    }


# ----------------------------------------------------------------------------- #
# 工况参数读取（用于 4.5.2 工况敏感性 / 4.5.3 失败案例归因）
# ----------------------------------------------------------------------------- #
# 关键索引依据 src/dataset/dataset_fast.py 中 _process_condition_normalize 的注释：
#   thermal[3] = P (W), thermal[4] = r0 (m), thermal[5] = domain length (m),
#   thermal[7] = A,     thermal[8] = v (m/s)
_THERMAL_IDX = {"P": 3, "r0": 4, "L": 5, "A": 7, "v": 8}


def get_condition_params(h5_path: str,
                         layer_thickness: float = 3.0e-5) -> Dict[str, float]:
    """
    从 h5 文件读取激光功率 / 有效半径 / 吸收率 / 扫描速度 / 计算域长度，并估算
    体积能量密度 E_v = A * P / (v * r0^2 * h)，其中 h 为层厚（默认 30 um）。
    """
    with h5py.File(h5_path, "r") as f:
        thermal = f["parameter/thermal"][:].reshape(-1)
    P = float(thermal[_THERMAL_IDX["P"]])
    r0 = float(thermal[_THERMAL_IDX["r0"]])
    L = float(thermal[_THERMAL_IDX["L"]])
    A = float(thermal[_THERMAL_IDX["A"]])
    v = float(thermal[_THERMAL_IDX["v"]])

    # 体积能量密度 E_v：单位 J/mm^3。这里把 r0 视为光斑等效半径，
    # h 取层厚（默认 30 um），与 ch4.md 4.5.2 节定义一致。
    eps = 1e-12
    E_v_si = A * P / (v * (r0 ** 2) * layer_thickness + eps)  # J/m^3
    E_v = E_v_si * 1e-9  # J/m^3 -> J/mm^3
    # 线性能量密度 E_l = P / v (J/m)，作为辅助度量
    E_l = P / (v + eps)
    return {
        "P": P, "r0": r0, "L": L, "A": A, "v": v,
        "E_v": float(E_v),
        "E_l": float(E_l),
    }


def sample_idx_to_h5(predictor: AeroGtoPredictor, idx: int) -> str:
    """根据样本索引回溯到原始 h5 文件路径。"""
    file_id, _ = predictor.dataset.sample_keys[idx]
    return predictor.dataset.file_paths[file_id]


__all__ = [
    "load_predictor",
    "rollout_sample",
    "per_sample_metrics",
    "get_condition_params",
    "sample_idx_to_h5",
]
