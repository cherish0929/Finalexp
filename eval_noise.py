"""
eval_noise.py — 输入噪声鲁棒性评估脚本
============================================================
以 evaluate_contrast.py 为模板，针对噪声鲁棒性实验：
  - 使用 test 模式 DataLoader（完整测试集 1/4 子集加速）；
  - 在初始状态 state[:, 0] 注入不同幅值的高斯白噪声：
    噪声标准差 = noise_level × 通道标准差（归一化空间 std ≈ 1）；
  - 对每个 noise_level 做完整 rollout，与干净 GT 对比；
  - 记录与 evaluate_contrast.py 一致的精度指标：
        mean_l2 / per-field L2 / per-field RMSE
  - 输出 clean-vs-noisy degradation ratio：
        Δ = (L2_noisy − L2_clean) / L2_clean × 100%
  - 额外记录效率指标：参数量、推理时间、GPU 显存峰值；
  - 结果输出到 result/noise/report_{timestamp}.txt

用法:
    编辑 CONFIG_LIST 和 NOISE_LEVELS，然后
        python eval_noise.py

Author: contrast experiment extension
============================================================
"""

import torch
import numpy as np
import os
import sys
import math
import time
import traceback
from datetime import datetime

from src.utils import load_json_config, set_seed
from src.model import build_model, GRID_MODELS


# ============================================================
# 需要评估的 config 路径列表（填写需要评估的模型配置）
# ============================================================
CONFIG_LIST = [
    "config/keyhole_scale/other_models/GTO_attnres_3_kh_scale_s.json",
    "config/keyhole_scale/other_models/GTO_kh_scale_s.json"
]

# 噪声幅值列表（以归一化空间通道标准差为单位，0.0 = 干净基线）
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05]


# ============================================================
# DualLogger — 同时打印到终端和文件
# ============================================================
class DualLogger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.terminal = sys.stdout

    def log(self, msg=""):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# DataLoader 创建（支持 grid_shape）
# ============================================================

def get_dataloader_eval(args, device_type):
    """创建 test DataLoader，额外返回 grid_shape 供网格模型使用。"""
    from torch.utils.data import DataLoader, Subset
    from src.dataset import AeroGtoDataset, CutAeroGtoDataset

    data_cfg = args.data

    if data_cfg.get("cut", False):
        Datasetclass = CutAeroGtoDataset
    else:
        Datasetclass = AeroGtoDataset
        
    args.data["test_list"], step = ["./data/con_ep/eval.txt"], 4
    args.data["test"]["batchsize"] = 1

    train_dataset = Datasetclass(args=args, mode="train")

    test_dataset = Datasetclass(
        args=args, mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None,
    )
    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()

    # 使用 1/4 测试集加速（与 evaluate_contrast.py 保持一致）
    step = 4
    subset_size = max(1, len(test_dataset) // step)
    indices = list(range(0, len(test_dataset), step))[:subset_size]
    test_dataset = Subset(test_dataset, indices)

    pin_memory = "cuda" in device_type
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_cfg["test"].get("batchsize", 1),
        shuffle=False,
        num_workers=data_cfg["test"].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    cond_dim    = args.model.get("cond_dim") or train_dataset.cond_dim
    default_dt  = train_dataset.dt

    sample_batch = next(iter(test_dataloader))
    grid_shape   = tuple(sample_batch["grid_shape"].tolist()[0])

    return test_dataloader, train_dataset.normalizer, cond_dim, default_dt, grid_shape


# ============================================================
# 模型加载（支持 grid_shape）
# ============================================================

def load_model_and_checkpoint(args, device, cond_dim, default_dt, grid_shape):
    """实例化模型并加载 best checkpoint + EMA shadow。"""
    model_cfg  = args.model
    model_name = model_cfg.get("name")

    if model_name in GRID_MODELS and grid_shape is None:
        return None, f"Model '{model_name}' requires grid_shape but none was provided."

    model = build_model(model_cfg, cond_dim, default_dt, device, grid_shape=grid_shape)

    ckpt_path = os.path.join(args.save_path, "nn", f"{args.name}_best.pt")
    if not os.path.exists(ckpt_path):
        return None, f"Checkpoint not found: {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "ema_shadow" in checkpoint:
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = model.state_dict()
        for k in ema_shadow:
            if k in state_dict:
                state_dict[k] = ema_shadow[k]
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    return model, {
        "path":           ckpt_path,
        "epoch":          checkpoint.get("epoch", "N/A"),
        "best_val_error": checkpoint.get("best_val_error", "N/A"),
    }


# ============================================================
# 打印 config 关键信息（与 evaluate_contrast.py 一致）
# ============================================================

def print_config_info(logger, config_path, args):
    data_cfg  = args.data
    model_cfg = args.model
    train_cfg = args.train

    logger.log(f"  Config file     : {config_path}")
    logger.log(f"  Experiment name : {args.name}")
    logger.log(f"  Save path       : {args.save_path}")

    # Model
    logger.log(f"  Model           : {model_cfg.get('name', 'PhysGTO')}")
    logger.log(f"  space_size={model_cfg.get('space_size', 3)}, "
               f"in_dim={model_cfg.get('in_dim', '?')}, out_dim={model_cfg.get('out_dim', '?')}, "
               f"enc_dim={model_cfg.get('enc_dim', 128)}, N_block={model_cfg.get('N_block', 4)}, "
               f"n_head={model_cfg.get('n_head', 4)}, n_token={model_cfg.get('n_token', 64)}")
    if model_cfg.get("n_fields"):
        logger.log(f"  n_fields={model_cfg['n_fields']}, cross_attn_heads={model_cfg.get('cross_attn_heads', 4)}")
    if model_cfg.get("attn_res_mode"):
        logger.log(f"  attn_res_mode   : {model_cfg['attn_res_mode']}")
    if model_cfg.get("stepper_scheme"):
        logger.log(f"  stepper_scheme  : {model_cfg['stepper_scheme']}")

    # Data
    logger.log(f"  Fields          : {data_cfg.get('fields', ['T'])}")
    logger.log(f"  Horizon (train) : {data_cfg.get('horizon_train', '?')}")
    logger.log(f"  Horizon (test)  : {data_cfg.get('horizon_test', '?')}")
    logger.log(f"  spatial_stride  : {data_cfg.get('spatial_stride', 'N/A')}")
    logger.log(f"  cut             : {data_cfg.get('cut', False)}")
    logger.log(f"  normalize       : {data_cfg.get('normalize', False)}")
    logger.log(f"  train_list      : {data_cfg.get('train_list', [])}")
    logger.log(f"  test_list       : {data_cfg.get('test_list', [])}")
    if data_cfg.get("active_mask"):
        logger.log(f"  active_mask     : {data_cfg['active_mask']}")

    # Train
    logger.log(f"  epoch           : {train_cfg.get('epoch', '?')}")
    logger.log(f"  lr              : {train_cfg.get('lr', '?')}")
    logger.log(f"  weight_decay    : {train_cfg.get('weight_decay', 'default')}")
    logger.log(f"  grad_loss_weight: {train_cfg.get('grad_loss_weight', 8.0)}")
    logger.log(f"  grad_clip       : {train_cfg.get('grad_clip', 'N/A')}")
    logger.log(f"  use_amp         : {train_cfg.get('use_amp', False)}")
    logger.log(f"  check_point     : {train_cfg.get('check_point', False)}")

    sched = train_cfg.get("scheduler", {})
    logger.log(f"  scheduler       : {sched.get('type', 'cosine')} "
               f"(warmup={sched.get('warmup_epochs', 'N/A')}, eta_min_ratio={sched.get('eta_min_ratio', 'N/A')})")

    wl = train_cfg.get("weight_loss", {})
    if wl.get("enable", False):
        logger.log(f"  weight_loss     : field={wl.get('field')}, thresh={wl.get('threshold')}, "
                   f"base_w={wl.get('base_weight')}, focus_w={wl.get('focus_weight')}")
    else:
        logger.log(f"  weight_loss     : disabled")

    pf = train_cfg.get("pushforward", {})
    if pf.get("enable", False):
        logger.log(f"  pushforward     : start={pf.get('start_epoch')}, extra={pf.get('extra_steps')}, ramp={pf.get('ramp_epochs')}")
    else:
        logger.log(f"  pushforward     : disabled")


# ============================================================
# 辅助：相对 L2 / RMSE
# ============================================================

def _relative_l2(pred, gt, eps=1e-8):
    """pred/gt: [B, T, N, C] 或 [B, N, C]。"""
    num = torch.sqrt(torch.sum((pred - gt) ** 2))
    den = torch.sqrt(torch.sum(gt ** 2)) + eps
    return (num / den).item()


def _rmse(pred, gt):
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


# ============================================================
# 单样本 noisy rollout
# ============================================================

def run_noisy_rollout(model, batch, device, args, normalizer, noise_level, rng):
    """
    对 batch 中的样本，将 state[:, 0] 加入高斯噪声后做完整 rollout，
    返回 (pred_real, gt_real) 均在物理空间（denormalized）。

    noise_level: 相对于通道标准差的比例。
    由于 state 已经过 normalizer 归一化（std ≈ 1 per channel），
    直接在归一化空间中注入 N(0, noise_level²) 噪声即等价于 noise_level × σ_channel。
    """
    model_name  = args.model.get("name", "PhysGTO")
    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max")

    dt         = batch["dt"].to(device)
    state      = batch["state"].to(device)          # [B, T, N, C]
    node_pos   = batch["node_pos"].to(device)
    edges      = batch["edges"].to(device)
    time_seq   = batch["time_seq"].to(device)
    conditions = batch["conditions"].to(device).float()
    if _use_spatial:
        spatial_inform = batch["spatial_inform"].to(device)

    # 注入噪声（归一化空间，std = noise_level）
    state_in = state[:, 0].clone()  # [B, N, C]
    if noise_level > 0.0:
        noise = torch.tensor(
            rng.normal(0.0, noise_level, state_in.shape),
            dtype=state_in.dtype, device=device
        )
        state_in = state_in + noise

    with torch.no_grad():
        if _use_spatial:
            predict = model.autoregressive(
                state_in, node_pos, edges, time_seq,
                spatial_inform, conditions, dt, False
            )
        else:
            predict = model.autoregressive(
                state_in, node_pos, edges, time_seq,
                conditions, dt, False
            )
        # predict: [B, T, N, C] 对应 time_seq 的 T 步；
        # state[:, 1:] 同样是 T 步 GT rollout，二者已对齐无需额外切片
        pred_seq = predict                 # [B, T, N, C]
        gt_seq   = state[:, 1:]            # [B, T, N, C]

    pred_real = normalizer.denormalize(pred_seq.float())
    gt_real   = normalizer.denormalize(gt_seq.float())
    return pred_real, gt_real


# ============================================================
# 单个 config 的噪声鲁棒性实验
# ============================================================

def evaluate_single_noise(config_path, device, logger):
    """
    遍历 NOISE_LEVELS，对测试集上的每个样本做 noisy rollout，
    汇总精度指标 + 效率指标 + clean-vs-noisy degradation。
    返回 dict 或 None（失败时）。
    """
    args = load_json_config(config_path)
    args.device = str(device)
    fields = args.data.get("fields", ["T"])

    if args.seed is not None:
        set_seed(args.seed)

    logger.log(f"\n{'='*70}")
    logger.log(f"Evaluating (noise robustness): {args.name}")
    logger.log(f"{'='*70}")

    print_config_info(logger, config_path, args)
    logger.log(f"  {'- '*35}")
    logger.log(f"  Noise levels    : {NOISE_LEVELS}")

    # ── 加载数据 ──────────────────────────────────────────────────
    logger.log("  Loading dataset ...")
    test_dataloader, normalizer, cond_dim, default_dt, grid_shape = \
        get_dataloader_eval(args, str(device))
    logger.log(f"  Test samples    : {len(test_dataloader.dataset)}, "
               f"batches: {len(test_dataloader)}")
    logger.log(f"  Grid shape      : {grid_shape}")

    # ── 加载模型 ──────────────────────────────────────────────────
    logger.log("  Loading model + checkpoint ...")
    model, ckpt_info = load_model_and_checkpoint(args, device, cond_dim, default_dt, grid_shape)
    if model is None:
        logger.log(f"  [SKIP] {ckpt_info}")
        return None
    logger.log(f"  Checkpoint      : {ckpt_info['path']} (epoch={ckpt_info['epoch']})")
    logger.log(f"  best_val_error  : {ckpt_info['best_val_error']}")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    logger.log(f"  #params (M)     : {params_m:.2f}")

    model.eval()
    normalizer.to(device)

    rng = np.random.default_rng(seed=42)

    noise_results = {}
    clean_l2_per_field = {}  # 记录 noise=0 时的 L2，用于 degradation 计算

    for noise_level in NOISE_LEVELS:
        logger.log(f"\n  -- noise_level = {noise_level:.2f} --")

        # 每个噪声水平重置 rng，保证不同 level 之间独立可复现
        level_rng = np.random.default_rng(seed=42 + int(noise_level * 1000))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        accum = {f: {"l2": [], "rmse": []} for f in fields}
        t0 = time.time()

        for batch in test_dataloader:
            try:
                pred_real, gt_real = run_noisy_rollout(
                    model, batch, device, args, normalizer, noise_level, level_rng
                )
            except Exception as e:
                logger.log(f"    [batch FAIL] {e}")
                continue

            # 累积每个字段的误差（在所有 (T-1) 步上）
            C = pred_real.shape[-1]
            for i, fname in enumerate(fields):
                p = pred_real[..., i:i+1]
                g = gt_real[..., i:i+1]
                accum[fname]["l2"].append(_relative_l2(p, g))
                accum[fname]["rmse"].append(_rmse(p, g))

        elapsed = time.time() - t0
        peak_mem = (torch.cuda.max_memory_allocated() / 1024 ** 2
                    if torch.cuda.is_available() else None)

        if all(len(accum[f]["l2"]) == 0 for f in fields):
            logger.log("    [WARN] no batches succeeded")
            continue

        entry = {}
        per_l2 = []
        for fname in fields:
            l2_vals  = accum[fname]["l2"]
            rmse_vals = accum[fname]["rmse"]
            if l2_vals:
                entry[f"L2_{fname}"]   = float(np.mean(l2_vals))
                entry[f"RMSE_{fname}"] = float(np.mean(rmse_vals))
                per_l2.append(entry[f"L2_{fname}"])
            else:
                entry[f"L2_{fname}"]   = float("nan")
                entry[f"RMSE_{fname}"] = float("nan")
        entry["mean_l2"]      = float(np.mean(per_l2)) if per_l2 else float("nan")
        entry["peak_mem_mb"]  = peak_mem
        entry["eval_time_s"]  = elapsed

        noise_results[noise_level] = entry

        # 记录 clean baseline 的 per-field L2
        if noise_level == 0.0:
            for fname in fields:
                clean_l2_per_field[fname] = entry.get(f"L2_{fname}", float("nan"))

        logger.log(f"    mean_l2         : {entry['mean_l2']:.6e}")
        for fname in fields:
            logger.log(f"    {fname:16s}: L2={entry[f'L2_{fname}']:.6e}, "
                       f"RMSE={entry[f'RMSE_{fname}']:.6e}")
        if peak_mem is not None:
            logger.log(f"    peak_memory (MB): {peak_mem:.1f}")
        logger.log(f"    eval_time (s)   : {elapsed:.1f}")

    # ── 计算 degradation（相对 clean baseline） ─────────────────
    clean_entry = noise_results.get(0.0, None)
    if clean_entry is not None:
        logger.log(f"\n  --- Degradation vs Clean Baseline (Δ = (noisy − clean) / clean × 100%) ---")
        for noise_level in NOISE_LEVELS:
            if noise_level == 0.0:
                continue
            entry = noise_results.get(noise_level)
            if entry is None:
                continue
            deg_mean = ((entry["mean_l2"] - clean_entry["mean_l2"])
                        / clean_entry["mean_l2"] * 100) if clean_entry["mean_l2"] != 0 else float("nan")
            entry["deg_mean_l2_pct"] = deg_mean
            logger.log(f"    σ={noise_level:.2f}:  mean_l2 Δ = {deg_mean:+.2f}%")
            for fname in fields:
                cl = clean_entry.get(f"L2_{fname}", float("nan"))
                nl = entry.get(f"L2_{fname}", float("nan"))
                deg = (nl - cl) / cl * 100 if (not math.isnan(cl) and cl != 0) else float("nan")
                entry[f"deg_L2_{fname}_pct"] = deg
                logger.log(f"      {fname:16s}: Δ = {deg:+.2f}%")

    return {
        "name":               args.name,
        "fields":             fields,
        "params_m":           params_m,
        "ckpt_best":          ckpt_info["best_val_error"],
        "noise_results":      noise_results,
        "clean_l2_per_field": clean_l2_per_field,
    }


# ============================================================
# 汇总表格
# ============================================================

def print_summary_table(logger, results):
    if not results:
        logger.log("\nNo successful evaluations.")
        return

    # ── 主表：各噪声水平 mean_l2 ─────────────────────────────────
    logger.log(f"\n{'='*100}")
    logger.log("SUMMARY — Noise Robustness (mean_l2 per noise level)")
    logger.log(f"{'='*100}")

    header_parts = [f"{'Name':30s}"]
    for nl in NOISE_LEVELS:
        header_parts.append(f"{'σ='+str(nl):>12s}")
    header_parts.append(f"{'params(M)':>10s}")
    header = " | ".join(header_parts)
    logger.log(header)
    logger.log("-" * len(header))

    for r in results:
        row = [f"{r['name']:30s}"]
        for nl in NOISE_LEVELS:
            entry = r["noise_results"].get(nl)
            if entry is not None and not math.isnan(entry.get("mean_l2", float("nan"))):
                row.append(f"{entry['mean_l2']:12.4e}")
            else:
                row.append(f"{'N/A':>12s}")
        row.append(f"{r['params_m']:10.2f}")
        logger.log(" | ".join(row))

    # ── Degradation 表（相对 clean） ─────────────────────────────
    nonzero_noises = [nl for nl in NOISE_LEVELS if nl > 0.0]
    if nonzero_noises:
        logger.log(f"\n{'='*100}")
        logger.log("SUMMARY — Degradation vs Clean Baseline (Δ%)")
        logger.log(f"{'='*100}")

        header_parts = [f"{'Name':30s}"]
        for nl in nonzero_noises:
            header_parts.append(f"{'σ='+str(nl):>12s}")
        header = " | ".join(header_parts)
        logger.log(header)
        logger.log("-" * len(header))

        for r in results:
            row = [f"{r['name']:30s}"]
            for nl in nonzero_noises:
                entry = r["noise_results"].get(nl)
                deg = entry.get("deg_mean_l2_pct") if entry else None
                if deg is not None and not math.isnan(deg):
                    row.append(f"{deg:+10.2f}%")
                else:
                    row.append(f"{'N/A':>12s}")
            logger.log(" | ".join(row))

    # ── Per-field L2 表 ──────────────────────────────────────────
    all_fields = []
    for r in results:
        for f in r["fields"]:
            if f not in all_fields:
                all_fields.append(f)

    for fname in all_fields:
        logger.log(f"\n  -- Field: {fname} (L2) --")
        hdr = [f"{'Name':30s}"] + [f"{'σ='+str(nl):>12s}" for nl in NOISE_LEVELS]
        logger.log(" | ".join(hdr))
        logger.log("-" * len(" | ".join(hdr)))
        for r in results:
            row = [f"{r['name']:30s}"]
            for nl in NOISE_LEVELS:
                entry = r["noise_results"].get(nl)
                val = entry.get(f"L2_{fname}") if entry else None
                if val is not None and not math.isnan(val):
                    row.append(f"{val:12.4e}")
                else:
                    row.append(f"{'N/A':>12s}")
            logger.log(" | ".join(row))

        # Per-field degradation
        if nonzero_noises:
            logger.log(f"    Degradation (%):")
            d_hdr = [f"{'Name':30s}"] + [f"{'σ='+str(nl):>12s}" for nl in nonzero_noises]
            logger.log("    " + " | ".join(d_hdr))
            logger.log("    " + "-" * len(" | ".join(d_hdr)))
            for r in results:
                row = [f"{r['name']:30s}"]
                for nl in nonzero_noises:
                    entry = r["noise_results"].get(nl)
                    deg = entry.get(f"deg_L2_{fname}_pct") if entry else None
                    if deg is not None and not math.isnan(deg):
                        row.append(f"{deg:+10.2f}%")
                    else:
                        row.append(f"{'N/A':>12s}")
                logger.log("    " + " | ".join(row))

    # ── Per-field RMSE 表 ────────────────────────────────────────
    for fname in all_fields:
        logger.log(f"\n  -- Field: {fname} (RMSE) --")
        hdr = [f"{'Name':30s}"] + [f"{'σ='+str(nl):>12s}" for nl in NOISE_LEVELS]
        logger.log(" | ".join(hdr))
        logger.log("-" * len(" | ".join(hdr)))
        for r in results:
            row = [f"{r['name']:30s}"]
            for nl in NOISE_LEVELS:
                entry = r["noise_results"].get(nl)
                val = entry.get(f"RMSE_{fname}") if entry else None
                if val is not None and not math.isnan(val):
                    row.append(f"{val:12.4e}")
                else:
                    row.append(f"{'N/A':>12s}")
            logger.log(" | ".join(row))


# ============================================================
# Main
# ============================================================

def main():
    if not CONFIG_LIST:
        print("ERROR: CONFIG_LIST is empty. Edit eval_noise.py and add config paths.")
        return

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"evaluate/noise/report_{timestamp}.txt"
    logger      = DualLogger(report_path)

    logger.log("#" * 70)
    logger.log("# Noise Robustness Evaluation Report")
    logger.log(f"# Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"# Device      : {device_str}")
    logger.log(f"# Configs     : {len(CONFIG_LIST)}")
    logger.log(f"# Noise levels: {NOISE_LEVELS}")
    logger.log(f"# Report      : {report_path}")
    logger.log("#" * 70)

    results, failures = [], []

    for idx, cfg in enumerate(CONFIG_LIST):
        logger.log(f"\n>>> [{idx+1}/{len(CONFIG_LIST)}] {cfg}")
        if not os.path.exists(cfg):
            logger.log("  [FAIL] config file not found")
            failures.append((cfg, "config missing"))
            continue
        try:
            agg = evaluate_single_noise(cfg, device, logger)
            if agg is not None:
                results.append(agg)
            else:
                failures.append((cfg, "skipped (checkpoint not found)"))
        except Exception as e:
            tb = traceback.format_exc()
            logger.log(f"  [FAIL] {e}\n{tb}")
            failures.append((cfg, str(e)))
        torch.cuda.empty_cache()

    print_summary_table(logger, results)

    if failures:
        logger.log(f"\n{'='*70}\nFAILURES ({len(failures)}):\n{'='*70}")
        for cfg, reason in failures:
            logger.log(f"  {cfg}\n    {reason}")

    logger.log("\n" + "#" * 70)
    logger.log(f"# Done. {len(results)} succeeded, {len(failures)} failed.")
    logger.log(f"# Report: {report_path}")
    logger.log("#" * 70)
    logger.close()


if __name__ == "__main__":
    main()
