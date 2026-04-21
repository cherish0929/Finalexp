"""
main_contrast.py — 对比实验训练主入口
============================================================
以 main_v2.py 为模板扩展，支持所有 10 个对比模型 + 本文模型。

与 main_v2.py 的区别:
  1. get_dataloader 额外返回 grid_shape（供 FNO3D / UNet3D 等）
  2. get_model 接受 grid_shape 并传给 build_model
  3. build_model 支持 9 个对比模型（MGN / GNOT / Transolver / GraphViT /
     FNO3D / UNet3D / MeltPoolResNet / ConvLSTMModel / ResNet3DModel）
  4. 其余训练 / 验证 / 保存逻辑与 main_v2.py **完全一致**

用法:
    python main_contrast.py --config config/contrast/MGN.json

Author: contrast experiment extension
============================================================
"""

import torch
import numpy as np
import os
import time
import copy
import math
import traceback
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from src.dataset import AeroGtoDataset, CutAeroGtoDataset
from src.train import train, validate, get_train_loss, _init_region_agg, _accumulate_region, _finalize_region
from src.utils import set_seed, init_weights, parse_args, load_json_config
from src.model import build_model, GRID_MODELS


# =============================================================================
# Error logging helpers  (与 main_v2.py 完全相同)
# =============================================================================

def _write_error(path_record, args_name, exc, context="", config_path=""):
    """Write exception info to both training_log.txt and a dedicated bug.txt."""
    timestamp = time.asctime(time.localtime(time.time()))
    tb_str = traceback.format_exc()

    # ---- training_log.txt (brief) ----
    try:
        log_path = f"{path_record}/{args_name}_training_log.txt"
        with open(log_path, "a") as f:
            f.write(f"\n{'='*20} ERROR {'='*20}\n")
            f.write(f"Time: {timestamp}\n")
            if context:
                f.write(f"Context: {context}\n")
            f.write(f"{tb_str}\n")
    except Exception:
        pass  # avoid recursive error

    # ---- bug.txt (detailed) ----
    try:
        bug_path = f"{path_record}/bug.txt"
        with open(bug_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Time      : {timestamp}\n")
            f.write(f"Config    : {config_path}\n")
            f.write(f"Run name  : {args_name}\n")
            if context:
                f.write(f"Context   : {context}\n")
            f.write(f"Exception : {type(exc).__name__}: {exc}\n")
            f.write(f"Traceback :\n{tb_str}\n")
    except Exception:
        pass


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """Linear warmup + monotonic cosine decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.eta_min = eta_min
        self.current_epoch = 0
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min
        )

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            self.cosine_scheduler.step()

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "cosine_scheduler": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state):
        self.current_epoch = state["current_epoch"]
        self.cosine_scheduler.load_state_dict(state["cosine_scheduler"])


# =============================================================================
# Pushforward Training  (与 main_v2.py 完全相同)
# =============================================================================

def train_pushforward(args, model, train_dataloader, optim, device, normalizer, extra_steps, ema=None, ckpt_threshold=None):
    """
    Pushforward training: extend rollout beyond horizon_train by extra_steps.
    Only backpropagate through the extra steps (the model is already good at the original horizon).
    This forces the model to learn to correct its own errors.
    """
    from torch.amp import autocast

    base_horizon = args.data.get("horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp = args.train.get("use_amp", False)
    # ckpt_threshold: False = config disabled; True/int = probe result; None = fallback to config
    if ckpt_threshold is not None:
        check_point = ckpt_threshold
    else:
        check_point = args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})
    grad_loss_weight = args.train.get("grad_loss_weight", 8.0)
    model_name = args.model.get("name", "PhysGTO")

    agg = {}
    for key in ["loss", "L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(base_horizon, device=device)
    agg["num"] = 0
    agg["value_loss"] = 0.0
    agg["grad_loss"] = 0.0
    has_region = False

    model.train()
    normalizer.to(device)

    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max")

    pbar = tqdm(train_dataloader, desc="  Train(PF)", unit="bt", leave=True, ncols=120, colour='cyan')

    for batch in pbar:
        dt = batch['dt'].to(device)
        state_cpu = batch["state"]           # [B, T+1, N, C] — stay on CPU until T_pf known
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq_cpu = batch["time_seq"]     # [B, T, 1] — stay on CPU until T_pf known
        if _use_spatial:
            spatial_inform = batch["spatial_inform"].to(device)
        conditions = batch["conditions"].to(device).float()
        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch['grid_shape'].numpy()

        active_mask = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask[:, 1:].to(device)
            if not has_region:
                _init_region_agg(agg, fields)
                has_region = True

        batch_num = state_cpu.shape[0]
        T_total = time_seq_cpu.shape[1]
        T_pf = min(base_horizon + extra_steps, T_total)

        # Transfer only needed slices to device
        state = state_cpu[:, :T_pf + 1].to(device)
        time_seq = time_seq_cpu[:, :T_pf].to(device)

        # Slice active_mask for base horizon
        base_mask = active_mask[:, :base_horizon] if active_mask is not None else None

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if _use_spatial:
                    predict_hat = model.autoregressive(
                        state[:, 0], node_pos, edges, time_seq[:, :T_pf], spatial_inform, conditions, dt, check_point)
                else:
                    predict_hat = model.autoregressive(
                        state[:, 0], node_pos, edges, time_seq[:, :T_pf], conditions, dt, check_point)
                # Loss on original horizon
                costs = get_train_loss(fields, predict_hat[:, :base_horizon], state[:, 1:base_horizon+1], normalizer, weight_loss, active_mask=base_mask)
                loss_base = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

                # Loss on extra steps (pushforward)
                if T_pf > base_horizon and state.shape[1] > base_horizon + 1:
                    T_extra = min(T_pf, state.shape[1] - 1)
                    pf_mask = active_mask[:, base_horizon:T_extra] if active_mask is not None else None
                    costs_pf = get_train_loss(
                        fields, predict_hat[:, base_horizon:T_extra],
                        state[:, base_horizon+1:T_extra+1], normalizer, weight_loss, active_mask=pf_mask
                    )
                    loss_pf = costs_pf["value_loss"] + grad_loss_weight * costs_pf["grad_loss"]
                    total_loss = loss_base + 0.5 * loss_pf  # pushforward weighted less
                else:
                    total_loss = loss_base

            # NaN guard / Loss spike guard: 跳过异常 batch，并释放计算图显存
            _skip = False
            if not torch.isfinite(total_loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and total_loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss_base, total_loss
                if 'costs_pf' in locals(): del costs_pf, loss_pf
                torch.cuda.empty_cache()
                continue

            # bf16 autocast without GradScaler
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            # Skip step if any gradient is non-finite
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                optim.zero_grad()
                del predict_hat, costs, loss_base, total_loss
                if 'costs_pf' in locals(): del costs_pf, loss_pf
                torch.cuda.empty_cache()
                continue
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)
        else:
            if _use_spatial:
                predict_hat = model.autoregressive(
                    state[:, 0], node_pos, edges, time_seq[:, :T_pf], spatial_inform, conditions, dt, check_point
                )
            else:
                predict_hat = model.autoregressive(
                    state[:, 0], node_pos, edges, time_seq[:, :T_pf], conditions, dt, check_point
                )
            costs = get_train_loss(fields, predict_hat[:, :base_horizon], state[:, 1:base_horizon+1], normalizer, weight_loss, active_mask=base_mask)
            loss_base = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

            if T_pf > base_horizon and state.shape[1] > base_horizon + 1:
                T_extra = min(T_pf, state.shape[1] - 1)
                pf_mask = active_mask[:, base_horizon:T_extra] if active_mask is not None else None
                costs_pf = get_train_loss(
                    fields, predict_hat[:, base_horizon:T_extra],
                    state[:, base_horizon+1:T_extra+1], normalizer, weight_loss, active_mask=pf_mask
                )
                loss_pf = costs_pf["value_loss"] + grad_loss_weight * costs_pf["grad_loss"]
                total_loss = loss_base + 0.5 * loss_pf
            else:
                total_loss = loss_base

            # NaN guard / Loss spike guard
            _skip = False
            if not torch.isfinite(total_loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and total_loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss_base, total_loss
                if 'costs_pf' in locals(): del costs_pf, loss_pf
                torch.cuda.empty_cache()
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            # Skip step if any gradient is non-finite
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                optim.zero_grad()
                del predict_hat, costs, loss_base, total_loss
                if 'costs_pf' in locals(): del costs_pf, loss_pf
                torch.cuda.empty_cache()
                continue
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)

        agg["loss"] += costs["loss"].item() * batch_num
        agg["value_loss"] += costs["value_loss"].item() * batch_num
        agg["grad_loss"] += costs["grad_loss"].item() * batch_num

        for fname in fields:
            agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
            agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
        agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
        agg["each_l2"] += costs["each_l2"] * batch_num
        agg["num"] += batch_num

        if has_region:
            _accumulate_region(agg, costs, batch_num, fields, include_loss=True)

        avg_loss = agg["loss"] / agg["num"]
        pbar.set_postfix({"Loss": f"{avg_loss:.4e}"})

    from src.train import _REGION_PREFIXES, _REGION_MEANS
    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in ("active_loss", "inactive_loss") and key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=True)
    return agg


# =============================================================================
# Patched train function with configurable grad_loss_weight  (与 main_v2.py 完全相同)
# =============================================================================

def train_v2(args, model, train_dataloader, optim, device, normalizer, ema=None, ckpt_threshold=None):
    """train() with configurable grad_loss_weight instead of hardcoded 8.0"""
    from torch.amp import autocast

    horizon = args.data.get("horizon_train", 1) if isinstance(args.data, dict) else getattr(args, "horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp = args.train.get("use_amp", False)
    # ckpt_threshold overrides config check_point when provided (set by _probe_ckpt_threshold)
    check_point = ckpt_threshold if ckpt_threshold is not None else args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})
    grad_loss_weight = args.train.get("grad_loss_weight", 8.0)
    model_name = args.model.get("name", "PhysGTO")

    agg = {}
    for key in ["loss", "L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0
    agg["value_loss"] = 0.0
    agg["grad_loss"] = 0.0
    has_region = False

    model.train()
    normalizer.to(device)

    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max")

    pbar = tqdm(train_dataloader, desc="  Train", unit="bt", leave=True, ncols=120, colour='green')
    for batch in pbar:
        dt = batch['dt'].to(device)
        state = batch["state"][:, :horizon + 1].to(device)
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq = batch["time_seq"][:, :horizon].to(device)
        if _use_spatial:
            spatial_inform = batch["spatial_inform"].to(device)
        conditions = batch["conditions"].to(device).float()
        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch['grid_shape'].numpy()

        active_mask = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask[:, 1:horizon + 1].to(device)
            if not has_region:
                _init_region_agg(agg, fields)
                has_region = True

        batch_num = state.shape[0]

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if _use_spatial:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
                else:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

                costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)

            # Use configurable grad_loss_weight
            loss = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

            # NaN guard / Loss spike guard: 跳过异常 batch，并释放计算图显存
            _skip = False
            if not torch.isfinite(loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue

            # bf16 autocast without GradScaler — more stable than fp16+scaler
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            # Skip step if any gradient is non-finite (prevents corrupting weights)
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)
        else:
            if _use_spatial:
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
            else:
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

            costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)

            loss = costs["value_loss"] + grad_loss_weight * costs["grad_loss"]

            # NaN guard / Loss spike guard
            _skip = False
            if not torch.isfinite(loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            # Skip step if any gradient is non-finite
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)

        # For logging, store the combined loss into costs
        costs["loss"] = loss

        agg["loss"] += costs["loss"].item() * batch_num
        agg["value_loss"] += costs["value_loss"].item() * batch_num
        agg["grad_loss"] += costs["grad_loss"].item() * batch_num

        for fname in fields:
            agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
            agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
        agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
        agg["each_l2"] += costs["each_l2"] * batch_num
        agg["num"] += batch_num

        if has_region:
            _accumulate_region(agg, costs, batch_num, fields, include_loss=True)

        avg_loss = agg["loss"] / agg["num"]
        pbar.set_postfix({"Loss": f"{avg_loss:.4e}"})

    from src.train import _REGION_PREFIXES, _REGION_MEANS
    if agg["num"] == 0:
        raise RuntimeError(
            "train_v2: all batches were skipped (NaN/spike guard triggered every batch). "
            "Loss is NaN or Inf — check for gradient explosion or bad input data."
        )
    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in ("active_loss", "inactive_loss") and key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=True)
    return agg


# =============================================================================
# Adaptive checkpoint threshold probing  (与 main_v2.py 完全相同)
# =============================================================================

def _probe_ckpt_threshold(model, first_batch, device, horizon, args, path_record="", run_name=""):
    """
    Binary-search for the maximum number of autoregressive steps that can run
    without gradient checkpointing before OOM.
    """
    if not torch.cuda.is_available():
        return False
    if horizon <= 1:
        return False

    from torch.amp import autocast

    use_amp = args.train.get("use_amp", False)
    model_name = args.model.get("name", "PhysGTO")
    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max")

    # Prepare a single-sample probe batch (use first sample only to minimise memory)
    dt = first_batch['dt'][:1].to(device)
    state = first_batch["state"][:1, :horizon + 1].to(device)
    node_pos = first_batch["node_pos"][:1].to(device)
    edges = first_batch["edges"][:1].to(device)
    time_seq = first_batch["time_seq"][:1, :horizon].to(device)
    conditions = first_batch["conditions"][:1].to(device).float()
    spatial_inform = None
    if _use_spatial and "spatial_inform" in first_batch:
        spatial_inform = first_batch["spatial_inform"][:1].to(device)

    def _try(ckpt_val):
        """Run forward+backward with given check_point value. Returns True on success."""
        torch.cuda.empty_cache()
        try:
            model.train()
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    if _use_spatial and spatial_inform is not None:
                        out = model.autoregressive(state[:, 0], node_pos, edges, time_seq,
                                                   spatial_inform, conditions, dt, ckpt_val)
                    else:
                        out = model.autoregressive(state[:, 0], node_pos, edges, time_seq,
                                                   conditions, dt, ckpt_val)
            else:
                if _use_spatial and spatial_inform is not None:
                    out = model.autoregressive(state[:, 0], node_pos, edges, time_seq,
                                               spatial_inform, conditions, dt, ckpt_val)
                else:
                    out = model.autoregressive(state[:, 0], node_pos, edges, time_seq,
                                               conditions, dt, ckpt_val)
            loss = out.float().sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            del out, loss
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False
        except Exception:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    def _log(msg):
        print(msg)
        if path_record and run_name:
            try:
                with open(f"{path_record}/{run_name}_training_log.txt", "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

    _log(f"[CKP Probe] Probing checkpoint threshold (horizon={horizon}) ...")

    # Step 1: try without any checkpointing
    if _try(False):
        _log(f"[CKP Probe] No OOM without ckp → ckpt_threshold={horizon} (no ckp used)")
        return horizon  # no ckp needed

    # Step 2: try with full checkpointing
    if not _try(True):
        _log(f"[CKP Probe] OOM even with full ckp → falling back to config check_point")
        return True  # let training handle it (will likely OOM, but that's the user's config)

    # Step 3: binary search for max k such that _try(k) succeeds
    lo, hi = 0, horizon - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _try(mid):
            lo = mid
        else:
            hi = mid - 1

    _log(f"[CKP Probe] ckpt_threshold={lo}: steps [0,{lo}) free, steps [{lo},{horizon}) use ckp")
    return lo


# =============================================================================
# DataLoader  (基于 main_v2.py，额外返回 grid_shape)
# =============================================================================

def _worker_init_fn(worker_id, base_seed):
    """固定每个 DataLoader worker 的随机种子，确保 shuffle 顺序可复现。"""
    import random as _random
    seed = base_seed + worker_id
    np.random.seed(seed)
    _random.seed(seed)
    torch.manual_seed(seed)


def _resolve_dataset_for_grouping(ds):
    """剥离 Subset，返回 (base_dataset, parent_index_map)。
    parent_index_map[i] = base_dataset 中对应原始下标。"""
    if isinstance(ds, Subset):
        base, idx_map = _resolve_dataset_for_grouping(ds.dataset)
        mapped = [idx_map[i] for i in ds.indices]
        return base, mapped
    return ds, list(range(len(ds)))


def _sample_node_count(base_dataset, sample_idx: int) -> int:
    """O(1) 取样本节点数：直接从 meta_cache 中查询，避免触发 __getitem__。"""
    file_id, _ = base_dataset.sample_keys[sample_idx]
    path = base_dataset.file_paths[file_id]
    return base_dataset.meta_cache[path]["node_pos"].shape[0]


class GridGroupedBatchSampler(Sampler):
    """按节点数（grid 形状）分组的 BatchSampler。

    Why: easypool 数据集中不同 HDF5 文件的网格形状不同，导致样本节点数不一致，
    PyTorch 默认 collate 在 batchsize>1 时无法 stack 变长 tensor。
    本采样器保证同一 batch 内的样本节点数一致，从而绕过 collate 的 stack 限制，
    且不引入 padding（不会污染 loss / 指标）。

    drop_last=False 时若某组余样本数 < batch_size，会形成一个小 batch（与默认行为一致）。
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool, seed=None, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)

        base, idx_map = _resolve_dataset_for_grouping(dataset)
        # 按节点数分组：{node_num: [dataloader 用的下标列表]}
        groups: dict = {}
        for i, parent_idx in enumerate(idx_map):
            n = _sample_node_count(base, parent_idx)
            groups.setdefault(n, []).append(i)
        self._groups = groups
        self._epoch = 0

    def _build_batches(self):
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + self._epoch
        )
        batches = []
        for _, idx_list in self._groups.items():
            order = list(idx_list)
            if self.shuffle:
                order = list(rng.permutation(order))
            for s in range(0, len(order), self.batch_size):
                chunk = order[s:s + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if self.shuffle:
            perm = rng.permutation(len(batches))
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        batches = self._build_batches()
        self._epoch += 1
        for b in batches:
            yield b

    def __len__(self):
        # 不依赖 epoch 状态：用一次稳定的计算
        total = 0
        for idx_list in self._groups.values():
            n = len(idx_list)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def get_dataloader(args, path_record, device_type, pf_extra_max=0):
    """
    创建训练/测试 DataLoader。
    与 main_v2.py 的区别：额外返回 grid_shape（供 FNO3D / UNet3D 等网格模型使用）。
    """
    data_cfg = args.data
    model_cfg = args.model
    space_dim = model_cfg.get("space_size", 3)

    if space_dim == 3:
        if data_cfg.get("cut", False):
            Datasetclass = CutAeroGtoDataset
        else:
            Datasetclass = AeroGtoDataset

    # Inject pf_extra_max into data_cfg so train dataset can load extra time steps
    if pf_extra_max > 0:
        data_cfg["horizon_pf_extra"] = pf_extra_max

    train_dataset = Datasetclass(
        args=args,
        mode="train"
    )

    test_dataset = Datasetclass(
        args=args,
        mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None
    )

    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()  # 同步 norm_mean/norm_std 缓存

    # Use 1/4 of the test set to reduce evaluation time
    subset_size = max(1, len(test_dataset) // 4)
    indices = list(range(0, len(test_dataset), 4))[:subset_size]
    test_dataset = Subset(test_dataset, indices)

    pin_memory = True if "cuda" in device_type else False

    seed = getattr(args, "seed", None)
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        dl_kwargs = dict(
            generator=g,
            worker_init_fn=lambda wid: _worker_init_fn(wid, seed),
        )
    else:
        dl_kwargs = {}

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=GridGroupedBatchSampler(
            train_dataset,
            batch_size=data_cfg['train'].get("batchsize", 1),
            shuffle=True,
            seed=seed,
            drop_last=False,
        ),
        num_workers=data_cfg['train'].get("num_workers", 0),
        pin_memory=pin_memory,
        worker_init_fn=dl_kwargs.get("worker_init_fn"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=GridGroupedBatchSampler(
            test_dataset,
            batch_size=data_cfg['test'].get("batchsize", 1),
            shuffle=False,
            seed=seed,
            drop_last=False,
        ),
        num_workers=data_cfg['test'].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    cond_dim = args.model.get("cond_dim") or train_dataset.cond_dim
    edge_num = train_dataset.meta_cache[train_dataset.file_paths[0]]["edges"].shape[0]

    # 从数据集获取 grid_shape（所有样本应该相同）
    sample_batch = next(iter(train_dataloader))
    grid_shape = tuple(sample_batch["grid_shape"].tolist()[0])
    print(grid_shape)

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
        file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
        file.write(f"Node num: {train_dataset.node_num}, Edge num: {edge_num}, Cond dim: {cond_dim}\n")
        file.write(f"Grid shape: {grid_shape}\n")
        file.write(f"Mean dt: {train_dataset.dt:.4e}\n")

    return train_dataloader, test_dataloader, train_dataset.normalizer, cond_dim, train_dataset.dt, grid_shape


# =============================================================================
# Model loading — 支持 grid_shape
# =============================================================================

def get_model(args, device, cond_dim, default_dt, grid_shape):
    """
    实例化模型并加载 checkpoint。
    与 main_v2.py 的区别：接受 grid_shape 参数并传给 build_model。
    """
    model_cfg = args.model

    model = build_model(model_cfg, cond_dim, default_dt, device, grid_shape=grid_shape)

    load_path = model_cfg.get("load_path")
    checkpoint = None

    if load_path:
        model_path = os.path.join(load_path, f"{args.name}_best.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {model_path}")
    elif model_cfg.get("if_init", True):
        model.apply(init_weights)

    return model, checkpoint


# =============================================================================
# Main training loop — 与 main_v2.py 逻辑相同
# =============================================================================

def main(args, path_logs, path_nn, path_record, config_path=""):
    device_str = args.device
    if "cuda" in device_str and not torch.cuda.is_available():
        print("! Warning: CUDA not available, using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    EPOCH = int(args.train["epoch"])
    real_lr = float(args.train["lr"])
    fields = args.data.get("fields", ["T"])

    # Pushforward config
    pf_cfg = args.train.get("pushforward", {"enable": False})
    pf_enable = pf_cfg.get("enable", False)
    pf_start = pf_cfg.get("start_epoch", 80)
    pf_extra_max = pf_cfg.get("extra_steps", 3)
    pf_ramp = pf_cfg.get("ramp_epochs", 40)

    # DataLoader & normalizer（额外返回 grid_shape）
    train_dataloader, test_dataloader, normalizer, cond_dim, default_dt, grid_shape = get_dataloader(
        args, path_record, device_str,
        pf_extra_max=pf_extra_max if pf_enable else 0
    )

    # Model（传入 grid_shape）
    model, checkpoint = get_model(args, device, cond_dim, default_dt, grid_shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = int(sum([np.prod(p.size()) for p in model_parameters]))

    print(f"EPOCH: {EPOCH}, #params: {params/1e6:.2f}M")

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"Using device: {device}\n")
        file.write(f"{args.name}, #params: {params/1e6:.2f}M\n")
        file.write(f"EPOCH: {EPOCH}\n")
        file.write(f"Fields: {fields}\n")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{path_logs}/{args.name}_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Optimizer
    weight_decay = args.train.get("weight_decay", real_lr / 50.0)
    optimizer = AdamW(model.parameters(), lr=real_lr, weight_decay=weight_decay)

    # Scheduler
    sched_cfg = args.train.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")

    if sched_type == "cosine_warmrestart":
        warmup_epochs = sched_cfg.get("warmup_epochs", 10)
        eta_min_ratio = sched_cfg.get("eta_min_ratio", 0.05)
        eta_min = real_lr * eta_min_ratio
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup_epochs,
            total_epochs=EPOCH, eta_min=eta_min
        )
        print(f"Scheduler: WarmupCosine (warmup={warmup_epochs}, total={EPOCH}, eta_min={eta_min:.2e})")
    else:
        if EPOCH < 50:
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=real_lr)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=real_lr / 20.0)
        print(f"Scheduler: CosineAnnealingLR (T_max={EPOCH})")

    # EMA
    ema = EMA(model, decay=0.998)
    print("EMA enabled (decay=0.998)")

    if pf_enable:
        print(f"Pushforward: ON (start={pf_start}, extra_max={pf_extra_max}, ramp={pf_ramp})")

    # ---- Adaptive checkpoint threshold (only when config enables it) ----
    horizon_train = args.data.get("horizon_train", 1)
    cfg_ckpt = args.train.get("check_point", False)
    if not cfg_ckpt:
        # Config explicitly disabled checkpointing — skip probe entirely
        print(f"[CKP] check_point disabled in config, skipping probe.")
        ckpt_threshold = False
    else:
        try:
            _first_batch = next(iter(train_dataloader))
            ckpt_threshold = _probe_ckpt_threshold(
                model, _first_batch, device, horizon_train, args,
                path_record=path_record, run_name=args.name
            )
            del _first_batch
            torch.cuda.empty_cache()
        except Exception as _probe_exc:
            print(f"[CKP Probe] Probing failed ({_probe_exc}), falling back to config check_point.")
            ckpt_threshold = cfg_ckpt  # fall back to config value (True or int)

    # Resume
    start_epoch, best_val_error = 0, float("inf")

    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Restored optimizer state.")
        if 'scheduler' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("Restored scheduler state.")
            except Exception:
                print("Scheduler state incompatible, starting fresh scheduler.")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        if 'best_val_error' in checkpoint:
            best_val_error = checkpoint['best_val_error']
            print(f"Restored best val error: {best_val_error:.4e}")

    if start_epoch >= EPOCH:
        print(f"Warning: Start epoch {start_epoch} >= Total EPOCH {EPOCH}.")

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"Optimizer: AdamW (lr={real_lr}, wd={weight_decay})\n")
        file.write(f"Scheduler: {sched_type}\n")
        file.write(f"EMA: decay=0.998\n")
        file.write(f"Pushforward: enable={pf_enable}\n")
        file.write(f"grad_loss_weight: {args.train.get('grad_loss_weight', 8.0)}\n")
        file.write(f"AMP: bf16 autocast (no GradScaler)\n")
        file.write(f"ckpt_threshold: {ckpt_threshold} (horizon={horizon_train})\n")

    # ---- Early-stop on repeated NaN/skip failures ----
    consecutive_skip_errors = 0
    max_consecutive_skip_errors = args.train.get("max_consecutive_errors", 3)

    # ---- Warn if dt is dangerously small for euler + AMP ----
    model_cfg = args.model
    train_cfg = args.train
    if (model_cfg.get("name") == "PhysGTO_v2"
            and model_cfg.get("stepper_scheme", "euler") == "euler"
            and train_cfg.get("use_amp", False)
            and not args.data.get("dt_scale", False)):
        _warn_msg = (
            "[WARNING] Potential NaN risk: PhysGTO_v2 with euler stepper + use_amp=True "
            "but dt_scale=False. Raw physical dt (e.g. 5e-6) forces v_pred to be ~1/dt "
            "larger, which overflows bfloat16. Consider setting dt_scale=true or use_amp=false."
        )
        print(_warn_msg)
        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(_warn_msg + "\n")

    # Interrupt checkpoint handler
    def _save_interrupt_ckpt(interrupt_epoch):
        if not args.if_save:
            return None
        curtime = datetime.now().strftime("%m-%d_%H-%M")
        ckpt = {
            'epoch': interrupt_epoch,
            'state_dict': model.state_dict(),
            'ema_shadow': ema.shadow,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'best_val_error': best_val_error,
            'config': args
        }
        interrupt_path = f"{path_nn}/{args.name}_interrupt_ep{interrupt_epoch}_{curtime}.pt"
        torch.save(ckpt, interrupt_path)
        return interrupt_path

    # ==================== Training Loop ====================
    for epoch in range(start_epoch, EPOCH):
        start_time = time.time()

        # Determine if pushforward is active
        use_pushforward = False
        pf_extra = 0
        if pf_enable and epoch >= pf_start:
            progress = min(1.0, (epoch - pf_start) / max(1, pf_ramp))
            pf_extra = max(1, int(round(progress * pf_extra_max)))
            use_pushforward = True

        # Train
        try:
            if use_pushforward:
                train_error = train_pushforward(
                    args, model, train_dataloader, optimizer, device, normalizer,
                    extra_steps=pf_extra, ema=ema, ckpt_threshold=ckpt_threshold
                )
            else:
                train_error = train_v2(
                    args, model, train_dataloader, optimizer, device, normalizer,
                    ema=ema, ckpt_threshold=ckpt_threshold
                )
        except KeyboardInterrupt:
            interrupt_epoch = epoch + 1
            if getattr(ema, "backup", None):
                ema.restore(model)
            interrupt_path = _save_interrupt_ckpt(interrupt_epoch)
            if interrupt_path is not None:
                print(f"[INTERRUPTED] Ctrl+C at epoch {interrupt_epoch}/{EPOCH}. Saved: {interrupt_path}")
            else:
                print(f"[INTERRUPTED] Ctrl+C at epoch {interrupt_epoch}/{EPOCH}. args.if_save=False, checkpoint not saved.")
            with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                file.write(f"[INTERRUPTED] Ctrl+C at epoch {interrupt_epoch}/{EPOCH}\n")
                if interrupt_path is not None:
                    file.write(f"Saved interrupt checkpoint: {interrupt_path}\n")
                else:
                    file.write("args.if_save=False, checkpoint not saved.\n")
            writer.close()
            return
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            _write_error(path_record, args.name, e,
                         context=f"train epoch {epoch + 1}/{EPOCH} — CUDA OOM (fatal, stopping)",
                         config_path=config_path)
            print(f"[FATAL OOM] Epoch {epoch + 1}: CUDA out of memory. Training stopped.")
            print(f"Error details saved to: {path_record}/{args.name}_training_log.txt")
            print(f"Full bug report saved to: {path_record}/bug.txt")
            writer.close()
            return
        except RuntimeError as e:
            _write_error(path_record, args.name, e,
                         context=f"train epoch {epoch + 1}/{EPOCH}",
                         config_path=config_path)
            print(f"[ERROR] Epoch {epoch + 1} training failed: {e}. Skipping epoch.")
            scheduler.step()
            # Check if this is a repeated all-batches-skipped error
            if "all batches were skipped" in str(e):
                consecutive_skip_errors += 1
                if consecutive_skip_errors >= max_consecutive_skip_errors:
                    _fatal_msg = (
                        f"[FATAL] {consecutive_skip_errors} consecutive epochs with all batches skipped "
                        f"(NaN/Inf). Training aborted at epoch {epoch + 1}/{EPOCH}. "
                        f"Check for numerical instability (e.g. dt_scale, use_amp, lr)."
                    )
                    print(_fatal_msg)
                    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                        file.write(_fatal_msg + "\n")
                    writer.close()
                    return
            else:
                consecutive_skip_errors = 0
            continue
        except Exception as e:
            _write_error(path_record, args.name, e,
                         context=f"train epoch {epoch + 1}/{EPOCH}",
                         config_path=config_path)
            print(f"[ERROR] Epoch {epoch + 1} training failed: {e}. Skipping epoch.")
            scheduler.step()
            consecutive_skip_errors = 0
            continue

        consecutive_skip_errors = 0
        end_time = time.time()

        # EMA is now updated per-batch inside train/train_pushforward
        scheduler.step()
        if hasattr(scheduler, 'get_last_lr'):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        training_time = (end_time - start_time)

        train_loss = train_error['loss']
        train_mean_l2 = train_error['mean_l2']
        each_t_l2 = train_error['each_l2']

        log_str = f"Training, Epoch: {epoch + 1}/{EPOCH}, train Loss: {train_loss:.4e}, mean_l2: {train_mean_l2:.4e}"
        writer.add_scalar('lr/lr', current_lr, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('L2/train_mean_l2', train_mean_l2, epoch)

        l2_details, rmse_details, region_l2_details = [], [], []
        for fname in fields:
            l2_val = train_error[f"L2_{fname}"]
            rmse_val = train_error[f"RMSE_{fname}"]
            l2_details.append(f"{fname}: {l2_val:.4e}")
            rmse_details.append(f"{fname}: {rmse_val:.4e}")
            writer.add_scalar(f'L2/train_L2_{fname}', l2_val, epoch)
            writer.add_scalar(f'RMSE/train_RMSE_{fname}', rmse_val, epoch)

            for prefix in ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE"):
                rval = train_error.get(f"{prefix}_{fname}")
                if rval is not None and not math.isnan(rval):
                    writer.add_scalar(f'{prefix}/train_{prefix}_{fname}', rval, epoch)

            a_l2 = train_error.get(f"active_L2_{fname}")
            i_l2 = train_error.get(f"inactive_L2_{fname}")
            if a_l2 is not None:
                region_l2_details.append(f"{fname}: act={a_l2:.4e}, inact={i_l2:.4e}")

        for key in ("active_mean_l2", "inactive_mean_l2"):
            val = train_error.get(key)
            if val is not None and not math.isnan(val):
                writer.add_scalar(f'L2/train_{key}', val, epoch)
        for key in ("active_loss", "inactive_loss"):
            val = train_error.get(key)
            if val is not None:
                writer.add_scalar(f'Loss/train_{key}', val, epoch)

        print(log_str)
        value_loss = train_error.get("value_loss", 0)
        grad_loss = train_error.get("grad_loss", 0)
        print(f"value_loss:{value_loss} | grad_loss:{grad_loss}")
        print(f"L2 details: {', '.join(l2_details)}")
        print(f"RMSE details: {', '.join(rmse_details)}")
        if region_l2_details:
            print(f"Region L2: {', '.join(region_l2_details)}")
        print(f"each time step loss: {each_t_l2.tolist()}")
        pf_info = f", pushforward extra={pf_extra}" if use_pushforward else ""
        print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}{pf_info}")
        print("--------------")

        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(f"Training, epoch: {epoch + 1}/{EPOCH}\n")
            file.write(f"Train Loss: {train_loss:.4e}, mean_l2: {train_mean_l2:.4e}\n")
            file.write(f"L2 details: {', '.join(l2_details)}\n")
            file.write(f"RMSE details: {', '.join(rmse_details)}\n")
            if region_l2_details:
                file.write(f"Region L2: {', '.join(region_l2_details)}\n")
            file.write(f"each time step loss: {each_t_l2.tolist()}\n")
            file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}{pf_info}\n")

        # Validation
        eval_every = args.train.get("eval_every", 5)
        if (epoch + 1) % eval_every == 0 or epoch == 0 or (epoch + 1) == EPOCH:
            start_time = time.time()

            try:
                torch.cuda.empty_cache()
                ema.apply_shadow(model)
                test_error = validate(args, model, test_dataloader, device, normalizer, epoch + 1)
                ema.restore(model)
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                ema.restore(model)
                torch.cuda.empty_cache()
                _write_error(path_record, args.name, e,
                             context=f"validate epoch {epoch + 1}/{EPOCH} — CUDA OOM (fatal, stopping)",
                             config_path=config_path)
                print(f"[FATAL OOM] Epoch {epoch + 1} validation: CUDA out of memory. Training stopped.")
                print(f"Error details saved to: {path_record}/{args.name}_training_log.txt")
                print(f"Full bug report saved to: {path_record}/bug.txt")
                writer.close()
                return
            except KeyboardInterrupt:
                ema.restore(model)
                interrupt_epoch = epoch + 1
                interrupt_path = _save_interrupt_ckpt(interrupt_epoch)
                if interrupt_path is not None:
                    print(f"[INTERRUPTED] Ctrl+C at epoch {interrupt_epoch}/{EPOCH}. Saved: {interrupt_path}")
                else:
                    print(f"[INTERRUPTED] Ctrl+C at epoch {interrupt_epoch}/{EPOCH}. args.if_save=False, checkpoint not saved.")
                with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                    file.write(f"[INTERRUPTED] Ctrl+C at epoch {interrupt_epoch}/{EPOCH}\n")
                    if interrupt_path is not None:
                        file.write(f"Saved interrupt checkpoint: {interrupt_path}\n")
                    else:
                        file.write("args.if_save=False, checkpoint not saved.\n")
                writer.close()
                return
            except Exception as e:
                ema.restore(model)
                torch.cuda.empty_cache()
                _write_error(path_record, args.name, e,
                             context=f"validate epoch {epoch + 1}/{EPOCH}",
                             config_path=config_path)
                print(f"[ERROR] Epoch {epoch + 1} validation failed: {e}. Skipping validation.")
                continue

            end_time = time.time()
            val_time = (end_time - start_time)

            test_mean_l2 = test_error['mean_l2']
            test_each_t_l2 = test_error['each_l2']

            test_l2_details, test_rmse_details, test_region_l2_details = [], [], []
            writer.add_scalar('L2/test_mean_l2', test_mean_l2, epoch)

            for fname in fields:
                l2_val = test_error[f"L2_{fname}"]
                rmse_val = test_error[f"RMSE_{fname}"]
                test_l2_details.append(f"{fname}: {l2_val:.4e}")
                test_rmse_details.append(f"{fname}: {rmse_val:.4e}")
                writer.add_scalar(f'L2/test_L2_{fname}', l2_val, epoch)
                writer.add_scalar(f'RMSE/test_RMSE_{fname}', rmse_val, epoch)

                for prefix in ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE"):
                    rval = test_error.get(f"{prefix}_{fname}")
                    if rval is not None and not math.isnan(rval):
                        writer.add_scalar(f'{prefix}/test_{prefix}_{fname}', rval, epoch)

                a_l2 = test_error.get(f"active_L2_{fname}")
                i_l2 = test_error.get(f"inactive_L2_{fname}")
                if a_l2 is not None:
                    test_region_l2_details.append(f"{fname}: act={a_l2:.4e}, inact={i_l2:.4e}")

            for key in ("active_mean_l2", "inactive_mean_l2"):
                val = test_error.get(key)
                if val is not None and not math.isnan(val):
                    writer.add_scalar(f'L2/test_{key}', val, epoch)

            print("---Inference (EMA)---")
            print(f"Epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}")
            print(f"L2 details: {', '.join(test_l2_details)}")
            print(f"RMSE details: {', '.join(test_rmse_details)}")
            if test_region_l2_details:
                print(f"Region L2: {', '.join(test_region_l2_details)}")
            print(f"each time step loss: {test_each_t_l2.tolist()}")
            print(f"time pre test epoch/s:{val_time:.2f}")
            print("--------------")

            with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
                file.write(f"Inference(EMA), epoch: {epoch + 1}/{EPOCH}, test_mean_l2: {test_mean_l2:.4e}\n")
                file.write(f"L2 details: {', '.join(test_l2_details)}\n")
                file.write(f"RMSE details: {', '.join(test_rmse_details)}\n")
                if test_region_l2_details:
                    file.write(f"Region L2: {', '.join(test_region_l2_details)}\n")
                file.write(f"each time step loss: {test_each_t_l2.tolist()}\n")
                file.write(f"time pre test epoch/s:{val_time:.2f}\n")

            # Save Best
            if args.if_save and test_mean_l2 < best_val_error:
                best_val_error = test_mean_l2
                ckpt = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_error': best_val_error,
                    'config': args
                }
                torch.save(ckpt, f"{path_nn}/{args.name}_best.pt")
                print(f"  >> New best! test_mean_l2={best_val_error:.4e}")

        # Regular Save
        if (epoch + 1) % 50 == 0 or (epoch + 1) == EPOCH:
            if args.if_save:
                ckpt = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'learning_rate': current_lr,
                    'best_val_error': best_val_error,
                }
                torch.save(ckpt, f"{path_nn}/{args.name}_{epoch+1}.pt")

    writer.close()


if __name__ == "__main__":
    cli_args = parse_args()
    args = load_json_config(cli_args.config)

    print(args)

    path_logs = args.save_path + "/logs"
    path_nn = args.save_path + "/nn"
    path_record = args.save_path + "/record"

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)
    os.makedirs(path_nn, exist_ok=True)
    os.makedirs(path_record, exist_ok=True)

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"{'='*20} Start (main_contrast) {'='*20}\n")
        file.write(str(args) + "\n")
        file.write(f"Config file: {cli_args.config}\n")
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")

    if args.seed is not None:
        set_seed(args.seed)

    try:
        main(args, path_logs, path_nn, path_record, config_path=cli_args.config)
    except Exception as e:
        _write_error(path_record, args.name, e,
                     context="main training loop (fatal crash)",
                     config_path=cli_args.config)
        print(f"[FATAL ERROR] Training crashed: {e}")
        print(f"Error details saved to: {path_record}/{args.name}_training_log.txt")
        print(f"Full bug report saved to: {path_record}/bug.txt")
        raise
    finally:
        with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
            file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")

