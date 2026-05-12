"""
main_v2.py — Improved training pipeline
============================================================
Compared to main.py, adds:
1. Warmup + CosineAnnealingWarmRestarts scheduler
   - Warmup epochs linearly increase LR to avoid early oscillations
   - Multi-cycle cosine annealing (T_0, T_mult) helps escape local minima
2. More reasonable weight_decay (1e-4 vs 1e-5)
3. Configurable gradient loss weight (default 5.0 vs hardcoded 8.0)
4. Pushforward training: gradually increases rollout steps during training
   - Mitigates autoregressive error accumulation
5. EMA (Exponential Moving Average) model, used for validation
6. More frequent eval (every N epochs, configurable)
7. Supports resuming from a previous best checkpoint
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
import json
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from src.dataset import AeroGtoDataset, AeroGtoDataset2D, CutAeroGtoDataset, LPBFLaserDataset
from src.train import (
    train, validate, get_train_loss,
    _init_region_agg, _accumulate_region, _finalize_region,
    _REGION_PREFIXES, _REGION_MEANS,
    train_v2, train_pushforward,
)
from src.utils import set_seed, init_weights, parse_args, load_json_config, _resolve_model_path
from src.utils.ema import EMA
from src.utils.scheduler import WarmupCosineScheduler
from src.utils.logging import write_error
from src.utils.data import worker_init_fn
from src.utils.ckpt_probe import (
    _compute_probe_cache_key,
    _ckpt_table_to_json,
    _ckpt_table_from_json,
    _format_ckpt_table,
    _diff_key_inputs,
    _load_probe_cache,
    _save_probe_cache,
    probe_ckpt_threshold,
)

# Alias for backward-compatible internal use
_write_error = write_error
_worker_init_fn = worker_init_fn


# =============================================================================
# DataLoader
# =============================================================================

def get_dataloader(args, path_record, device_type, pf_extra_max=0):
    data_cfg = args.data
    model_cfg = args.model
    space_dim = model_cfg.get("space_size", 3)
    model_name = model_cfg.get("name", "PhysGTO")

    if model_name == "gto_lpbf":
        Datasetclass = LPBFLaserDataset
        model_cfg["_fields"] = data_cfg.get("fields", ["T"])
    elif space_dim == 3:
        if data_cfg.get("cut", False):
            Datasetclass = CutAeroGtoDataset
        else:
            Datasetclass = AeroGtoDataset
    else:  # space_dim == 2
        Datasetclass = AeroGtoDataset2D

    if pf_extra_max > 0:
        data_cfg["horizon_pf_extra"] = pf_extra_max

    train_dataset = Datasetclass(args=args, mode="train")

    test_dataset = Datasetclass(
        args=args,
        mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None
    )

    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()

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
        batch_size=data_cfg['train'].get("batchsize", 1),
        shuffle=True,
        num_workers=data_cfg['train'].get("num_workers", 0),
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_cfg['test'].get("batchsize", 1),
        shuffle=False,
        num_workers=data_cfg['test'].get("num_workers", 0),
        pin_memory=pin_memory,
    )

    cond_dim = args.model.get("cond_dim") or train_dataset.cond_dim
    edge_num = train_dataset.meta_cache[train_dataset.file_paths[0]]["edges"].shape[0]

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
        file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
        file.write(f"Node num: {train_dataset.node_num}, Edge num: {edge_num}, Cond dim: {cond_dim}\n")
        file.write(f"Mean dt: {train_dataset.dt:.4e}\n")

    return train_dataloader, test_dataloader, train_dataset.normalizer, cond_dim, train_dataset.dt


# =============================================================================
# Model loading
# =============================================================================

def get_model(args, device, cond_dim, default_dt):
    from src.model import build_model
    model_cfg = args.model

    model = build_model(model_cfg, cond_dim, default_dt, device)

    load_path = model_cfg.get("load_path")
    checkpoint = None

    if load_path:
        resolved_path = _resolve_model_path(load_path, args.name)
        if resolved_path and os.path.exists(resolved_path):
            try:
                checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
                state_dict = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded model from {resolved_path}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint {resolved_path!r}: {e}")
                print("Falling back to random initialization.")
                checkpoint = None
                if model_cfg.get("if_init", True):
                    model.apply(init_weights)
        else:
            print(f"Warning: no checkpoint found for load_path={load_path!r}, training from scratch")
            if model_cfg.get("if_init", True):
                model.apply(init_weights)
    elif model_cfg.get("if_init", True):
        model.apply(init_weights)

    return model, checkpoint


# =============================================================================
# Main training loop
# =============================================================================

def main(args, path_logs, path_nn, path_record, config_path=""):

    device_str = args.device
    data_cfg = args.data
    if "cuda" in device_str and not torch.cuda.is_available():
        print("! Warning: CUDA not available, using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    EPOCH = int(args.train["epoch"])
    real_lr = float(args.train["lr"])
    fields = args.data.get("fields", ["T"])

    # ---- Pushforward config ----
    pf_cfg = args.train.get("pushforward", {"enable": False})
    pf_enable = pf_cfg.get("enable", False)
    pf_start = pf_cfg.get("start_epoch", 80)
    pf_extra_max = pf_cfg.get("extra_steps", 3)
    pf_ramp = pf_cfg.get("ramp_epochs", 40)

    # Dataloader & normalizer
    train_dataloader, test_dataloader, normalizer, cond_dim, default_dt = get_dataloader(
        args, path_record, device_str,
        pf_extra_max=pf_extra_max if pf_enable else 0
    )

    # Model
    model, checkpoint = get_model(args, device, cond_dim, default_dt)
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

    # ---- Optimizer ----
    weight_decay = args.train.get("weight_decay", real_lr / 50.0)
    optimizer = AdamW(model.parameters(), lr=real_lr, weight_decay=weight_decay)

    # ---- Scheduler ----
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

    # ---- EMA ----
    ema = EMA(model, decay=0.998)
    print("EMA enabled (decay=0.998)")

    if pf_enable:
        print(f"Pushforward: ON (start={pf_start}, extra_max={pf_extra_max}, ramp={pf_ramp})")

    # ---- Adaptive checkpoint threshold ----
    horizon_train = args.data.get("horizon_train", 1)
    cfg_ckpt = args.train.get("check_point", False)
    probe_horizon = horizon_train + (pf_extra_max if pf_enable else 0)
    ckpt_table = None
    if not cfg_ckpt:
        print(f"[CKP] check_point disabled in config, skipping probe.")
    else:
        try:
            _ds = train_dataloader.dataset
            if hasattr(_ds, 'dataset'):
                _ds = _ds.dataset
            _train_bs = data_cfg['train'].get("batchsize", 1) if isinstance(data_cfg, dict) else 1

            _max_node_count = 0
            _max_path = None
            if hasattr(_ds, 'meta_cache') and hasattr(_ds, 'file_paths'):
                _max_path = max(_ds.file_paths, key=lambda p: _ds.meta_cache[p]["node_pos"].shape[0])
                _max_node_count = _ds.meta_cache[_max_path]["node_pos"].shape[0]

            # ---- Try cache hit ----
            _cache_path = f"{path_record}/{args.name}_ckpt_probe_cache.json"
            _cache_key, _key_inputs = _compute_probe_cache_key(args, _max_node_count, device)
            _cached_table, _old_key_inputs = _load_probe_cache(_cache_path, _cache_key)

            if _cached_table is not None:
                ckpt_table = _cached_table
                _hit_msg = (f"[CKP Cache] HIT (key={_cache_key}, "
                            f"file={_cache_path})\n[CKP Cache] Cached per-horizon table:\n"
                            f"{_format_ckpt_table(ckpt_table)}")
                print(_hit_msg)
                try:
                    with open(f"{path_record}/{args.name}_training_log.txt", "a") as _f:
                        _f.write(_hit_msg + "\n")
                except OSError:
                    pass
            else:
                if _old_key_inputs is not None:
                    print(f"[CKP Cache] MISS (cache key changed: "
                          f"{_diff_key_inputs(_old_key_inputs, _key_inputs)}), re-probing.")
                else:
                    print(f"[CKP Cache] MISS (no cache at {_cache_path}), probing fresh.")

                if _max_path is not None and hasattr(_ds, 'meta_cache') and hasattr(_ds, 'file_paths'):
                    _max_indices = [i for i, (fid, _) in enumerate(_ds.sample_keys)
                                    if _ds.file_paths[fid] == _max_path][:_train_bs]
                    if not _max_indices:
                        _max_indices = [next(i for i, (fid, _) in enumerate(_ds.sample_keys)
                                              if _ds.file_paths[fid] == _max_path)]
                    print(f"[CKP Probe] Using largest-mesh sample (path={_max_path}, "
                          f"N={_max_node_count}, probe_bs={len(_max_indices)}) for probe.")
                    _first_batch = train_dataloader.collate_fn([_ds[i] for i in _max_indices])
                else:
                    _first_batch = next(iter(train_dataloader))
                ckpt_table = probe_ckpt_threshold(
                    model, _first_batch, device, probe_horizon, args,
                    path_record=path_record, run_name=args.name,
                    optimizer=optimizer, normalizer=normalizer,
                )
                del _first_batch
                torch.cuda.empty_cache()

                if ckpt_table is not None:
                    _save_probe_cache(_cache_path, _cache_key, _key_inputs, ckpt_table)
                    print(f"[CKP Cache] Saved to {_cache_path}")
        except Exception as _probe_exc:
            print(f"[CKP Probe] Probing failed ({_probe_exc}), falling back to config check_point.")
            ckpt_table = None

    def _get_ckpt_threshold(actual_horizon):
        """Look up the optimal checkpoint threshold for a given rollout length."""
        if ckpt_table is None:
            return False if not cfg_ckpt else True
        if actual_horizon in ckpt_table:
            return ckpt_table[actual_horizon]
        closest = min(ckpt_table.keys(), key=lambda h: (h < actual_horizon, abs(h - actual_horizon)))
        if closest >= actual_horizon:
            return ckpt_table[closest]
        return ckpt_table.get(max(ckpt_table.keys()), True)

    # ---- Resume ----
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
        print(f"Warning: Start epoch {start_epoch} >= Total EPOCH {EPOCH}. Training may perform 0 steps.")

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as file:
        file.write(f"Optimizer: AdamW (lr={real_lr}, wd={weight_decay})\n")
        file.write(f"Scheduler: {sched_type}\n")
        file.write(f"EMA: decay=0.998\n")
        file.write(f"Pushforward: enable={pf_enable}\n")
        file.write(f"grad_loss_multiplier: {args.train.get('weight_loss', {}).get('grad_loss_multiplier', 8.0)}\n")
        file.write(f"AMP: bf16 autocast (no GradScaler)\n")
        file.write(f"ckpt_table: {ckpt_table} (probe_horizon={probe_horizon}, base={horizon_train})\n")

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
                actual_horizon = horizon_train + pf_extra
                ckpt_threshold = _get_ckpt_threshold(actual_horizon)
                train_error = train_pushforward(
                    args, model, train_dataloader, optimizer, device, normalizer,
                    extra_steps=pf_extra, ema=ema, ckpt_threshold=ckpt_threshold
                )
            else:
                ckpt_threshold = _get_ckpt_threshold(horizon_train)
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

        # Step scheduler
        scheduler.step()
        if hasattr(scheduler, 'get_last_lr'):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        training_time = (end_time - start_time)

        # Extract metrics
        train_loss = train_error['loss']
        train_mean_l2 = train_error['mean_l2']
        each_t_l2 = train_error['each_l2']

        # Log
        log_str = f"Training, Epoch: {epoch + 1}/{EPOCH}, train Loss: {train_loss:.4e}, mean_l2: {train_mean_l2:.4e}"
        writer.add_scalar('lr/lr', current_lr, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('L2/train_mean_l2', train_mean_l2, epoch)

        l2_details = []
        rmse_details = []
        region_l2_details = []
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

        # Validation (using EMA model)
        eval_every = args.train.get("eval_every", 5)
        if (epoch + 1) % eval_every == 0 or epoch == 0 or (epoch + 1) == EPOCH:
            start_time = time.time()

            try:
                torch.cuda.empty_cache()
                ema.apply_shadow(model)
                test_error = validate(args, model, test_dataloader, device, normalizer, epoch + 1)
                ema.restore(model)
                torch.cuda.empty_cache()
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

            test_l2_details = []
            test_rmse_details = []
            test_region_l2_details = []
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
        file.write(f"{'='*20} Start (main_v2) {'='*20}\n")
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
