"""
calibrate_weights.py — 权重系数诊断与校准脚本

用法:
    conda activate GTO
    python calibrate_weights.py --config config/easypool/GTO_attnres_3_easypool_stronger.json

功能:
    1. 读取配置文件（物理场、threshold 等均从配置中解析）
    2. 遍历训练集的若干 batch，对每个物理场统计：
       - 原始 MSE（无加权）
       - 活跃区 / 非活跃区的点数比例
       - 活跃区内 MSE vs 非活跃区内 MSE
       - 梯度损失量级（gradient loss per field）
    3. 基于统计量推导一组建议权重配置，并打印对比
"""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- 项目内部 ----
from src.dataset import AeroGtoDataset, CutAeroGtoDataset
from src.utils import load_json_config, ChannelNormalizer
from src.train import (
    compute_spatial_gradient_3d,
    _is_vof_field,
    _build_weight_mask,
)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _deep_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _load_args(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # 递归转 dict → SimpleNamespace 方便点号访问
    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_ns(x) for x in obj]
        return obj
    return _to_ns(cfg)


def _build_dataset(args):
    data_cfg = args.data if isinstance(args.data, dict) else vars(args.data)
    cut = data_cfg.get("cut", False)
    if cut:
        ds = CutAeroGtoDataset(args, mode="train")
    else:
        ds = AeroGtoDataset(args, mode="train")
    return ds


# ---------------------------------------------------------------------------
# 梯度损失计算（仅用于统计，不参与反向传播）
# ---------------------------------------------------------------------------

def _grad_mse_per_field(pred, gt, grid_shape):
    """
    Returns scalar MSE of spatial gradient for one field.
    pred, gt: [B, T, N, 1]
    grid_shape: [B, 3] numpy array
    """
    pg_x, pg_y, pg_z = compute_spatial_gradient_3d(pred, grid_shape)
    gg_x, gg_y, gg_z = compute_spatial_gradient_3d(gt, grid_shape)
    loss = (F.mse_loss(pg_x, gg_x) +
            F.mse_loss(pg_y, gg_y) +
            F.mse_loss(pg_z, gg_z))
    return loss.item()


# ---------------------------------------------------------------------------
# 主统计逻辑
# ---------------------------------------------------------------------------

class FieldStats:
    """在线累积器，计算均值（Welford 不动点）。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum_mse_all = 0.0       # 全部点 MSE
        self.sum_mse_active = 0.0    # 活跃区 MSE
        self.sum_mse_inactive = 0.0  # 非活跃区 MSE
        self.sum_grad_mse = 0.0      # 梯度 MSE
        self.active_ratio_sum = 0.0  # 活跃点比例累计
        self.focus_active_mse_sum = 0.0   # focus_weight 加权后 active MSE
        self.focus_inactive_mse_sum = 0.0

    def add(self, mse_all, mse_act, mse_inact, grad_mse, ratio, n_batch):
        self.n += n_batch
        self.sum_mse_all      += mse_all * n_batch
        self.sum_mse_active   += mse_act * n_batch
        self.sum_mse_inactive += mse_inact * n_batch
        self.sum_grad_mse     += grad_mse * n_batch
        self.active_ratio_sum += ratio * n_batch

    @property
    def mse_all(self):      return self.sum_mse_all / self.n if self.n else 0
    @property
    def mse_active(self):   return self.sum_mse_active / self.n if self.n else 0
    @property
    def mse_inactive(self): return self.sum_mse_inactive / self.n if self.n else 0
    @property
    def grad_mse(self):     return self.sum_grad_mse / self.n if self.n else 0
    @property
    def active_ratio(self): return self.active_ratio_sum / self.n if self.n else 0


def run_stats(args, max_batches: int = 30, device: str = "cpu"):
    fields     = args.data.get("fields", ["T"]) if isinstance(args.data, dict) else vars(args.data).get("fields", ["T"])
    weight_cfg = args.train.get("weight_loss", {}) if isinstance(args.train, dict) else vars(args.train).get("weight_loss", {})
    if not isinstance(weight_cfg, dict):
        weight_cfg = vars(weight_cfg)

    w_fields     = weight_cfg.get("field", [])
    thresholds   = weight_cfg.get("threshold", [])
    has_gradient = weight_cfg.get("gradient", False)

    print(f"\n[诊断] 物理场: {fields}")
    print(f"[诊断] 活跃区阈值配置: {list(zip(w_fields, thresholds))}")
    print(f"[诊断] 梯度损失启用: {has_gradient}")

    ds = _build_dataset(args)
    normalizer = ds.normalizer
    normalizer.to(device)

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    stats = {f: FieldStats() for f in fields}

    print(f"\n[统计] 最多处理 {max_batches} 个 batch（来自训练集），请稍候...\n")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        state_raw = batch["state"].to(device)        # [B, T, N, C]  归一化
        grid_shape = batch["grid_shape"].numpy()     # [B, 3]

        B, T_full, N, C = state_raw.shape
        # 用前 T-1 步作 "pred"（模拟零偏差 baseline），后 T-1 步作 GT
        # 目的：衡量各 loss 项的量级，不需要真实模型输出
        T = T_full - 1
        pred_norm = state_raw[:, :T]     # [B, T, N, C]  normalized (作为 "完美预测" 基准)
        gt_norm   = state_raw[:, 1:]     # [B, T, N, C]  normalized

        with torch.no_grad():
            pred_real = normalizer.denormalize(pred_norm.float())
            gt_real   = normalizer.denormalize(gt_norm.float())

        for fi, fname in enumerate(fields):
            fld_pred_n = pred_norm[..., fi:fi+1].float()
            fld_gt_n   = gt_norm[..., fi:fi+1].float()
            fld_gt_r   = gt_real[..., fi:fi+1]

            mse_all = F.mse_loss(fld_pred_n, fld_gt_n).item()

            # 活跃区 mask
            if fname in w_fields:
                i_thresh = w_fields.index(fname)
                thresh   = thresholds[i_thresh]
                with torch.no_grad():
                    if isinstance(thresh, list):
                        mask = (fld_gt_r > thresh[0]) & (fld_gt_r < thresh[1])
                    else:
                        mask = fld_gt_r > thresh

                ratio = mask.float().mean().item()
                total_pts = mask.numel()
                active_pts = mask.sum().item()

                if active_pts > 0:
                    mse_act = (((fld_pred_n - fld_gt_n) ** 2) * mask.float()).sum().item() / active_pts
                else:
                    mse_act = 0.0

                inact_pts = total_pts - active_pts
                if inact_pts > 0:
                    mse_inact = (((fld_pred_n - fld_gt_n) ** 2) * (~mask).float()).sum().item() / inact_pts
                else:
                    mse_inact = 0.0
            else:
                ratio    = 0.0
                mse_act  = mse_all
                mse_inact = mse_all

            # 梯度损失（仅在有 grid_shape 时计算）
            grad_mse = 0.0
            if has_gradient and grid_shape is not None:
                try:
                    grad_mse = _grad_mse_per_field(fld_pred_n, fld_gt_n, grid_shape)
                except Exception as e:
                    grad_mse = float('nan')

            stats[fname].add(mse_all, mse_act, mse_inact, grad_mse, ratio, B)

        sys.stdout.write(f"\r  已处理 {batch_idx+1}/{max_batches} batches")
        sys.stdout.flush()

    print("\n")
    return stats, fields, w_fields, thresholds, has_gradient


# ---------------------------------------------------------------------------
# 打印统计报告
# ---------------------------------------------------------------------------

def print_stats(stats, fields, w_fields, thresholds):
    sep = "=" * 72
    print(sep)
    print(f"{'字段':<14} {'全局MSE':>12} {'活跃区MSE':>12} {'非活MSE':>12} {'活跃占比':>10} {'梯度MSE':>12}")
    print("-" * 72)
    for fname in fields:
        s = stats[fname]
        print(f"{fname:<14} {s.mse_all:>12.4e} {s.mse_active:>12.4e} "
              f"{s.mse_inactive:>12.4e} {s.active_ratio:>10.2%} {s.grad_mse:>12.4e}")
    print(sep)


# ---------------------------------------------------------------------------
# 权重推荐算法
# ---------------------------------------------------------------------------

def recommend_weights(stats, fields, w_fields, thresholds, has_gradient):
    """
    推荐策略：
    1. focus_weight / base_weight:
       - 目标：让活跃区贡献 ~50% 的该场 value loss
       - 设 p = active_ratio, 求 fw/bw 使得 p*fw / (p*fw + (1-p)*bw) = 0.5
       - 解：fw/bw = (1-p)/p，取 bw=1 → fw = (1-p)/p
       - 若 p > 0.5（活跃区已超半），则 fw=bw=1（不需要额外聚焦）
       - clamp 到 [2, 100] 防止极端值

    2. 场间 value loss 平衡 (grad_weight):
       - 希望各场梯度 loss 量级相近
       - gw_i ∝ 1 / grad_mse_i（倒数归一化）
       - 再缩放使最小权重 = 1

    3. grad_loss_multiplier:
       - 让 grad_loss_total ≈ 0.3 × value_loss_total
       - multiplier = 0.3 × mean_value_mse / mean_grad_mse
       - clamp [0.1, 10]
    """
    rec = {}

    # ---------- 1. focus_weight ----------
    base_weights = []
    focus_weights = []
    for fname in fields:
        if fname not in w_fields:
            base_weights.append(None)
            focus_weights.append(None)
            continue
        s = stats[fname]
        p = s.active_ratio
        if p <= 0 or p >= 0.5:
            fw = 1.0
        else:
            fw = (1.0 - p) / p  # 让活跃区贡献 ~50%
        fw = float(np.clip(fw, 2.0, 100.0))
        base_weights.append(1.0)
        focus_weights.append(fw)

    rec["base_weight"] = [bw if bw is not None else 1.0 for bw in base_weights]
    rec["focus_weight"] = [fw if fw is not None else 1.0 for fw in focus_weights]

    # ---------- 2. grad_weight ----------
    if has_gradient:
        grad_mses = {}
        for fname in fields:
            gm = stats[fname].grad_mse
            if math.isnan(gm) or gm <= 0:
                gm = 1e-8
            grad_mses[fname] = gm

        # 倒数归一化：最大梯度 MSE 的场权重最小（=1），其余按比例放大
        max_grad = max(grad_mses.values())
        raw_gw = {f: max_grad / gm for f, gm in grad_mses.items()}
        # 归一化使最小权重 = 1
        min_gw = min(raw_gw.values())
        grad_weights = {f: round(v / min_gw, 2) for f, v in raw_gw.items()}
        rec["grad_weight"] = grad_weights

        # ---------- 3. grad_loss_multiplier ----------
        mean_value_mse = np.mean([stats[f].mse_all for f in fields])
        # grad_mse_total ≈ sum over fields （每个 field 乘上推荐 grad_weight）
        grad_total_est = sum(gm * grad_weights.get(f, 1.0) for f, gm in grad_mses.items())
        if grad_total_est > 0:
            multiplier = 0.3 * mean_value_mse * len(fields) / grad_total_est
            multiplier = float(np.clip(multiplier, 0.1, 10.0))
        else:
            multiplier = 1.0
        rec["grad_loss_multiplier"] = round(multiplier, 2)

    return rec


# ---------------------------------------------------------------------------
# 打印推荐配置
# ---------------------------------------------------------------------------

def print_recommendations(rec, fields, w_fields, thresholds, has_gradient,
                           orig_cfg: dict):
    sep = "=" * 72
    print(sep)
    print("推荐权重配置（可直接复制到 JSON）:")
    print(sep)

    # 构建可直接粘贴的 weight_loss dict
    suggested = {
        "enable": True,
        "field": w_fields,
        "threshold": thresholds,
        "base_weight": rec.get("base_weight", [1.0] * len(w_fields)),
        "focus_weight": rec.get("focus_weight", [5.0] * len(w_fields)),
    }

    if has_gradient:
        suggested["gradient"] = True
        suggested["grad_weight"] = rec.get("grad_weight", {f: 1.0 for f in fields})
        suggested["grad_loss_multiplier"] = rec.get("grad_loss_multiplier", 1.0)
        suggested["grad_weight_mode"] = "adaptive"  # 推荐 adaptive，比 mask 更鲁棒
        suggested["grad_focus_weight"] = orig_cfg.get("grad_focus_weight", [5.0, 8.0])
        suggested["grad_base_weight"] = orig_cfg.get("grad_base_weight", [1.0, 1.0])
        suggested["laplacian"] = False
        suggested["sharpness"] = False

    print(json.dumps({"weight_loss": suggested}, indent=2, ensure_ascii=False))
    print(sep)

    # 对比原始配置
    print("\n对比（原始 vs 推荐）:")
    print(f"  {'字段':<14}  {'原 base':>8}  {'推荐 base':>10}  {'原 focus':>10}  {'推荐 focus':>11}")
    print("  " + "-" * 60)
    orig_bw = orig_cfg.get("base_weight", [])
    orig_fw = orig_cfg.get("focus_weight", [])
    for i, fname in enumerate(w_fields):
        ob = orig_bw[i] if i < len(orig_bw) else "—"
        of = orig_fw[i] if i < len(orig_fw) else "—"
        rb = rec["base_weight"][i] if "base_weight" in rec and i < len(rec["base_weight"]) else "—"
        rf = rec["focus_weight"][i] if "focus_weight" in rec and i < len(rec["focus_weight"]) else "—"
        print(f"  {fname:<14}  {str(ob):>8}  {str(rb):>10}  {str(of):>10}  {str(rf):>11}")

    if has_gradient:
        orig_mult = orig_cfg.get("grad_loss_multiplier", "—")
        rec_mult  = rec.get("grad_loss_multiplier", "—")
        print(f"\n  grad_loss_multiplier: {orig_mult}  →  {rec_mult}")
        print(f"  grad_weight_mode:     {orig_cfg.get('grad_weight_mode','mask')}  →  adaptive（推荐）")
        print("\n  推荐 grad_weight（场间梯度平衡）:")
        for f, gw in rec.get("grad_weight", {}).items():
            orig_gw = orig_cfg.get("grad_weight", {}).get(f, "—")
            print(f"    {f}: {orig_gw}  →  {gw}")

    print(sep)
    print("\n提示:")
    print("  1. 本脚本使用「完美预测」基准（pred = prev_state）估计 loss 量级，")
    print("     实际训练中模型有误差，量级会更大，但比例关系基本稳定。")
    print("  2. 推荐 threshold 对 alpha/gamma 类 VOF 场使用更宽范围，如 [0.05, 0.95]，")
    print("     可有效增加活跃样本点数（当前配置可能过窄）。")
    print("  3. grad_weight_mode='adaptive' 用 |∇gt| 作为空间权重，")
    print("     对界面天然聚焦，比 mask 模式更稳定。")
    print("  4. 建议先用推荐值跑 5-10 epoch，观察 TensorBoard 中 value_loss 与")
    print("     grad_loss 的比例，再微调 grad_loss_multiplier。")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Loss weight calibration tool")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--batches", type=int, default=30, help="统计的 batch 数（默认30）")
    parser.add_argument("--device", type=str, default="cpu", help="统计设备（cpu 即可）")
    cli = parser.parse_args()

    print(f"\n{'='*72}")
    print(f" Loss 权重校准工具  |  配置: {cli.config}")
    print(f"{'='*72}")

    args = _load_args(cli.config)

    # 将 SimpleNamespace 属性转成 dict 以便兼容 dataset
    def _ns_to_dict(obj):
        if isinstance(obj, SimpleNamespace):
            return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
        if isinstance(obj, list):
            return [_ns_to_dict(x) for x in obj]
        return obj

    # dataset 期望 args.data / args.train 是 dict
    args_dict_data  = _ns_to_dict(args.data)
    args_dict_train = _ns_to_dict(args.train)

    class DictArgs:
        pass

    dargs = DictArgs()
    dargs.data  = args_dict_data
    dargs.train = args_dict_train

    orig_wl_cfg = args_dict_train.get("weight_loss", {})

    stats, fields, w_fields, thresholds, has_gradient = run_stats(
        dargs, max_batches=cli.batches, device=cli.device
    )

    print_stats(stats, fields, w_fields, thresholds)

    rec = recommend_weights(stats, fields, w_fields, thresholds, has_gradient)
    print_recommendations(rec, fields, w_fields, thresholds, has_gradient, orig_wl_cfg)


if __name__ == "__main__":
    main()
