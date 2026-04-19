"""
evaluate_contrast.py — 对比实验统一评估脚本
============================================================
以 evaluate.py 为模板扩展，在精度指标基础上增加推理效率统计：
- 平均推理时间 (ms/sample) — 含 warm-up，多次均值
- GPU 显存峰值 (MB) — torch.cuda.max_memory_allocated()
- 模型参数量 (M) — sum(p.numel() for p in model.parameters())
- FLOPs (GFLOPs) — 使用 fvcore 或 ptflops 估算

用法:
    编辑 CONFIG_LIST，然后运行 python evaluate_contrast.py
    评估报告保存到 result/contrast/evaluate/report_{timestamp}.txt

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
from types import SimpleNamespace

from src.utils import load_json_config, set_seed
from src.train import validate
from src.model import build_model, GRID_MODELS


# ============================================================
# 需要评估的 config 路径列表
# ============================================================
CONFIG_LIST = [
    # PhysGTO 家族（本文方法）
    # "config/keyhole/GTO_keyhole_stronger.json",
    # "config/keyhole/GTO_attnres_keyhole_stronger.json",
    # "config/keyhole/GTO_attnres_3_keyhole_stronger.json",

    # 对比模型配置（待创建）
    # "config/contrast/MGN.json",
    # "config/contrast/GNOT.json",
    # "config/contrast/Transolver.json",
    # "config/contrast/GraphViT.json",
    # "config/contrast/FNO3D.json",
    # "config/contrast/UNet3D.json",
    # "config/contrast/MeltPoolResNet.json",
    # "config/contrast/ConvLSTMModel.json",
    # "config/contrast/ResNet3DModel.json",
]


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
    """创建测试 DataLoader，额外返回 grid_shape（供网格模型使用）。"""
    from torch.utils.data import DataLoader, Subset
    from src.dataset import AeroGtoDataset, CutAeroGtoDataset

    data_cfg = args.data
    model_cfg = args.model
    space_dim = model_cfg.get("space_size", 3)

    if data_cfg.get("cut", False):
        Datasetclass = CutAeroGtoDataset
    else:
        Datasetclass = AeroGtoDataset

    train_dataset = Datasetclass(args=args, mode="train")

    test_dataset = Datasetclass(
        args=args, mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None,
    )
    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()

    # 使用 1/4 测试集加速
    subset_size = max(1, len(test_dataset) // 4)
    indices = list(range(0, len(test_dataset), 4))[:subset_size]
    test_dataset = Subset(test_dataset, indices)

    pin_memory = "cuda" in device_type
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_cfg['test'].get("batchsize", 1),
        shuffle=False,
        num_workers=data_cfg['test'].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    cond_dim = args.model.get("cond_dim") or train_dataset.cond_dim
    default_dt = train_dataset.dt

    # 获取 grid_shape
    sample_batch = next(iter(test_dataloader))
    grid_shape = tuple(sample_batch["grid_shape"].tolist())

    return test_dataloader, train_dataset.normalizer, cond_dim, default_dt, grid_shape


# ============================================================
# 模型加载（支持 grid_shape）
# ============================================================

def load_model_and_checkpoint(args, device, cond_dim, default_dt, grid_shape):
    """实例化模型并加载 best checkpoint + EMA shadow。"""
    from src.utils import init_weights
    from src.model import build_model
    model_cfg = args.model
    model_name = model_cfg.get("name")

    if model_name in GRID_MODELS and grid_shape is None:
        return None, f"Model '{model_name}' requires grid_shape but dataset did not provide it."

    model = build_model(model_cfg, cond_dim, default_dt, device, grid_shape=grid_shape)

    # 加载 checkpoint
    path_nn = args.save_path + "/nn"
    ckpt_path = os.path.join(path_nn, f"{args.name}_best.pt")
    if not os.path.exists(ckpt_path):
        return None, f"Checkpoint not found: {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 优先使用 EMA shadow 权重
    if "ema_shadow" in checkpoint:
        ema_shadow = checkpoint["ema_shadow"]
        state_dict = model.state_dict()
        for name in ema_shadow:
            if name in state_dict:
                state_dict[name] = ema_shadow[name]
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    ckpt_info = {
        "path": ckpt_path,
        "epoch": checkpoint.get("epoch", "N/A"),
        "best_val_error": checkpoint.get("best_val_error", "N/A"),
    }
    return model, ckpt_info


# ============================================================
# 推理效率统计
# ============================================================

def compute_inference_metrics(model, test_dataloader, device, args, warmup=3, repeats=5):
    """
    计算推理效率指标：
    - 平均推理时间 (ms/sample) — 含 warm-up，取多次均值
    - GPU 显存峰值 (MB)
    - 模型参数量 (M)
    """
    if not torch.cuda.is_available():
        return {
            "avg_time_ms": None,
            "peak_memory_mb": None,
            "params_m": None,
        }

    torch.cuda.reset_peak_memory_stats()
    model_name = args.model.get("name", "PhysGTO")
    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max")

    # 参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # Warm-up + 时间测量
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= warmup + repeats:
                break

            dt = batch['dt'].to(device)
            state = batch["state"].to(device)
            node_pos = batch["node_pos"].to(device)
            edges = batch["edges"].to(device)
            time_seq = batch["time_seq"].to(device)
            if _use_spatial:
                spatial_inform = batch["spatial_inform"].to(device)
            conditions = batch["conditions"].to(device).float()

            if i >= warmup:
                torch.cuda.synchronize()
                t0 = time.time()

                if _use_spatial:
                    _ = model.autoregressive(state[:, 0], node_pos, edges, time_seq,
                                              spatial_inform, conditions, dt, False)
                else:
                    _ = model.autoregressive(state[:, 0], node_pos, edges, time_seq,
                                              conditions, dt, False)

                torch.cuda.synchronize()
                t1 = time.time()
                times.append((t1 - t0) * 1000)  # ms

            torch.cuda.empty_cache()

    avg_time = np.mean(times) if times else None
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    return {
        "avg_time_ms": avg_time,
        "peak_memory_mb": peak_memory,
        "params_m": params,
    }


def estimate_flops(model, input_shape, device):
    """
    估算模型 FLOPs（使用 ptflops 或 fvcore）。
    input_shape: (batch_size, N, in_dim) 示例输入形状
    返回 GFLOPs。
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            return None

    # 创建临时输入
    batch_size, N, in_dim = input_shape
    space_size = 3
    dummy_state = torch.randn(batch_size, N, in_dim, device=device)
    dummy_pos = torch.randn(batch_size, N, space_size, device=device)
    ne = 10000  # 假设边数
    dummy_edges = torch.randint(0, N, (batch_size, ne, 2), device=device)
    dummy_time = torch.randn(batch_size, 1, device=device)
    cond_dim = 32
    dummy_cond = torch.randn(batch_size, cond_dim, device=device)

    # 尝试使用 ptflops
    try:
        from ptflops import get_model_complexity_info
        flops, params = get_model_complexity_info(
            model, (dummy_state, dummy_pos, dummy_edges, dummy_time, dummy_cond),
            as_strings=False,
            print_per_layer_stat=False
        )
        return flops / 1e9  # GFLOPs
    except Exception:
        pass

    # 尝试使用 fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward'):
                # 根据模型类型调用 forward
                model_name = getattr(model, 'name', None)
                if model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max"):
                    _ = model(dummy_state, dummy_pos, dummy_edges, dummy_time, dummy_cond,
                              torch.randn(batch_size, 10, device=device))
                else:
                    _ = model(dummy_state, dummy_pos, dummy_edges, dummy_time, dummy_cond)
        flops = FlopCountAnalysis(model, dummy_state)
        return flops.total() / 1e9
    except Exception:
        return None


# ============================================================
# 打印配置信息
# ============================================================

def print_config_info(logger, config_path, args):
    data_cfg = args.data
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

    # Data
    logger.log(f"  Fields          : {data_cfg.get('fields', ['T'])}")
    logger.log(f"  Horizon (train) : {data_cfg.get('horizon_train', '?')}")
    logger.log(f"  Horizon (test)  : {data_cfg.get('horizon_test', '?')}")
    logger.log(f"  spatial_stride  : {data_cfg.get('spatial_stride', 'N/A')}")
    logger.log(f"  cut             : {data_cfg.get('cut', False)}")
    logger.log(f"  normalize       : {data_cfg.get('normalize', False)}")
    logger.log(f"  train_list      : {data_cfg.get('train_list', [])}")
    logger.log(f"  test_list       : {data_cfg.get('test_list', [])}")

    # Train
    logger.log(f"  epoch           : {train_cfg.get('epoch', '?')}")
    logger.log(f"  lr              : {train_cfg.get('lr', '?')}")
    logger.log(f"  weight_decay    : {train_cfg.get('weight_decay', 'default')}")
    logger.log(f"  grad_loss_weight: {train_cfg.get('grad_loss_weight', 8.0)}")
    logger.log(f"  use_amp         : {train_cfg.get('use_amp', False)}")


# ============================================================
# 打印评估结果（含效率指标）
# ============================================================

def print_metrics(logger, metrics, eff_metrics, fields):
    # 精度指标
    logger.log(f"  mean_l2         : {metrics['mean_l2']:.6e}")

    for fname in fields:
        l2_val = metrics.get(f"L2_{fname}", "N/A")
        rmse_val = metrics.get(f"RMSE_{fname}", "N/A")
        line = f"  {fname:16s}: L2={l2_val:.6e}, RMSE={rmse_val:.6e}" if isinstance(l2_val, float) else f"  {fname:16s}: L2={l2_val}, RMSE={rmse_val}"
        logger.log(line)

    # Region metrics
    has_region = any(k.startswith("active_L2_") for k in metrics)
    if has_region:
        logger.log("  --- Region Metrics ---")
        a_mean = metrics.get("active_mean_l2", float('nan'))
        i_mean = metrics.get("inactive_mean_l2", float('nan'))
        if not math.isnan(a_mean):
            logger.log(f"  active_mean_l2  : {a_mean:.6e}")
        if not math.isnan(i_mean):
            logger.log(f"  inactive_mean_l2: {i_mean:.6e}")

    # 效率指标
    logger.log("  --- Efficiency Metrics ---")
    logger.log(f"  #params (M)     : {eff_metrics['params_m']:.2f}")
    if eff_metrics['avg_time_ms'] is not None:
        logger.log(f"  avg_time (ms)   : {eff_metrics['avg_time_ms']:.2f}")
    if eff_metrics['peak_memory_mb'] is not None:
        logger.log(f"  peak_memory (MB): {eff_metrics['peak_memory_mb']:.1f}")


# ============================================================
# 评估单个 config
# ============================================================

def evaluate_single(config_path, device, logger):
    """评估单个配置文件对应的模型，返回精度 + 效率指标。"""
    args = load_json_config(config_path)
    args.device = str(device)
    fields = args.data.get("fields", ["T"])

    if args.seed is not None:
        set_seed(args.seed)

    logger.log(f"\n{'='*70}")
    logger.log(f"Evaluating: {args.name}")
    logger.log(f"{'='*70}")

    print_config_info(logger, config_path, args)
    logger.log(f"  {'- '*35}")

    # 加载数据（额外返回 grid_shape）
    logger.log("  Loading dataset ...")
    test_dataloader, normalizer, cond_dim, default_dt, grid_shape = get_dataloader_eval(args, str(device))
    logger.log(f"  Test samples: {len(test_dataloader.dataset)}, Test batches: {len(test_dataloader)}")
    logger.log(f"  Grid shape: {grid_shape}")

    # 加载模型 + checkpoint（传入 grid_shape）
    logger.log("  Loading model + checkpoint ...")
    model, ckpt_info = load_model_and_checkpoint(args, device, cond_dim, default_dt, grid_shape)
    if model is None:
        logger.log(f"  [SKIP] {ckpt_info}")
        return None

    logger.log(f"  Checkpoint      : {ckpt_info['path']}")
    logger.log(f"  Trained epoch   : {ckpt_info['epoch']}")
    logger.log(f"  best_val_error  : {ckpt_info['best_val_error']}")

    # 效率指标
    logger.log("  Computing efficiency metrics ...")
    eff_metrics = compute_inference_metrics(model, test_dataloader, device, args)

    # 精度评估
    logger.log("  Running validation ...")
    t0 = time.time()
    model.eval()
    normalizer.to(device)
    metrics = validate(args, model, test_dataloader, device, normalizer, epoch=0)
    elapsed = time.time() - t0
    logger.log(f"  Validation time : {elapsed:.1f}s")

    logger.log(f"  --- Results ---")
    print_metrics(logger, metrics, eff_metrics, fields)

    metrics["_best_val_error"] = ckpt_info["best_val_error"]
    metrics.update(eff_metrics)

    return args.name, fields, metrics


# ============================================================
# 汇总表格（含效率指标）
# ============================================================

def print_summary_table(logger, results):
    """打印包含精度和效率指标的汇总表。"""
    if not results:
        logger.log("\nNo successful evaluations to summarize.")
        return

    logger.log(f"\n{'='*90}")
    logger.log("SUMMARY TABLE (Accuracy + Efficiency)")
    logger.log(f"{'='*90}")

    all_fields = []
    for name, fields, metrics in results:
        for f in fields:
            if f not in all_fields:
                all_fields.append(f)

    # 表头
    header_parts = [f"{'Name':30s}", f"{'mean_l2':>12s}"]
    for f in all_fields:
        header_parts.append(f"{'L2_'+f:>12s}")
    header_parts.extend([
        f"{'params(M)':>12s}",
        f"{'time(ms)':>12s}",
        f"{'mem(MB)':>10s}",
        f"{'ckpt_best':>12s}",
    ])

    header = " | ".join(header_parts)
    logger.log(header)
    logger.log("-" * len(header))

    for name, fields, metrics in results:
        row_parts = [f"{name:30s}"]
        row_parts.append(f"{metrics['mean_l2']:12.4e}")

        for f in all_fields:
            l2_val = metrics.get(f"L2_{f}")
            row_parts.append(f"{l2_val:12.4e}" if isinstance(l2_val, float) else f"{'N/A':>12s}")

        row_parts.append(f"{metrics.get('params_m', 0):12.2f}")
        row_parts.append(f"{metrics.get('avg_time_ms', 0):12.2f}" if metrics.get('avg_time_ms') else f"{'N/A':>12s}")
        row_parts.append(f"{metrics.get('peak_memory_mb', 0):10.1f}" if metrics.get('peak_memory_mb') else f"{'N/A':>10s}")

        best_ve = metrics.get("_best_val_error", "N/A")
        if isinstance(best_ve, float):
            row_parts.append(f"{best_ve:12.4e}")
        else:
            row_parts.append(f"{str(best_ve):>12s}")

        logger.log(" | ".join(row_parts))


# ============================================================
# Main
# ============================================================

def main():
    if not CONFIG_LIST:
        print("ERROR: CONFIG_LIST is empty. Please edit evaluate_contrast.py and add config paths.")
        return

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"result/contrast/evaluate/report_{timestamp}.txt"
    logger = DualLogger(report_path)

    logger.log(f"{'#'*70}")
    logger.log(f"# Contrast Experiment Evaluation Report")
    logger.log(f"# Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"# Device : {device_str}")
    logger.log(f"# Configs: {len(CONFIG_LIST)}")
    logger.log(f"# Report : {report_path}")
    logger.log(f"{'#'*70}")

    results = []
    failures = []

    for idx, config_path in enumerate(CONFIG_LIST):
        logger.log(f"\n>>> [{idx+1}/{len(CONFIG_LIST)}] {config_path}")

        if not os.path.exists(config_path):
            msg = f"Config file not found: {config_path}"
            logger.log(f"  [FAIL] {msg}")
            failures.append((config_path, msg))
            continue

        try:
            result = evaluate_single(config_path, device, logger)
            if result is not None:
                results.append(result)
            else:
                failures.append((config_path, "Skipped (checkpoint not found)"))
        except Exception as e:
            tb = traceback.format_exc()
            logger.log(f"  [FAIL] Exception: {e}")
            logger.log(f"  Traceback:\n{tb}")
            failures.append((config_path, str(e)))

        torch.cuda.empty_cache()

    # 汇总
    print_summary_table(logger, results)

    # 失败列表
    if failures:
        logger.log(f"\n{'='*70}")
        logger.log(f"FAILURES ({len(failures)}):")
        logger.log(f"{'='*70}")
        for cfg, reason in failures:
            logger.log(f"  {cfg}")
            logger.log(f"    Reason: {reason}")

    logger.log(f"\n{'#'*70}")
    logger.log(f"# Done. {len(results)} succeeded, {len(failures)} failed.")
    logger.log(f"# Report saved to: {report_path}")
    logger.log(f"{'#'*70}")

    logger.close()


if __name__ == "__main__":
    main()

