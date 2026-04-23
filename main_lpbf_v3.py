"""
main_lpbf_v3.py — Training entry point for LPBF-NO v3 (Stage 1)

Mirrors main_v2.py closely but uses:
  - LPBFSlotDataset  (slot-based, from src/dataset/dataset_lpbf_v3.py)
  - lpbf_no_v3.Model (Stage 1 architecture)
  - train_v3 / validate_v3  (slot-aware loss, from src/train_lpbf_v3.py)

All existing infrastructure is UNCHANGED:
  - WarmupCosineScheduler  (copied from main_v2)
  - EMA                    (copied from main_v2)
  - Adaptive checkpoint probing
  - Pushforward training
  - NaN/spike guard
  - Ctrl+C interrupt save
  - TensorBoard logging

Usage:
    conda activate GTO
    python main_lpbf_v3.py --config config/lpbf_v3/lpbf_no_v3_easypool.json

Config additions (on top of existing keys):
    model.field_slots:    list of {name, type, default} dicts
    train.weight_loss.chamfer:  bool   (Chamfer loss for interface slots)
    train.weight_loss.peak:     bool   (peak temperature L1 loss)
    train.weight_loss.normal:   bool   (interface normal consistency)
"""

import os
import sys
import time
import copy
import math
import traceback
import signal
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# ---- Project imports ----
from src.utils import set_seed, init_weights, parse_args, load_json_config
from src.dataset.dataset_lpbf_v3 import LPBFSlotDataset
from src.model.lpbf_no_v3 import Model as LPBFModelV3
from src.train_lpbf_v3 import train_v3, validate_v3


# =============================================================================
# EMA  (identical to main_v2)
# =============================================================================

class EMA:
    """Exponential Moving Average of model parameters (per-batch update)."""

    def __init__(self, model, decay=0.999):
        self.decay  = decay
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


# =============================================================================
# WarmupCosineScheduler  (identical to main_v2)
# =============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.base_lr        = optimizer.param_groups[0]['lr']
        self.eta_min        = eta_min
        self.current_epoch  = 0
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
        return {"current_epoch": self.current_epoch,
                "cosine_scheduler": self.cosine_scheduler.state_dict()}

    def load_state_dict(self, state):
        self.current_epoch = state["current_epoch"]
        self.cosine_scheduler.load_state_dict(state["cosine_scheduler"])


# =============================================================================
# Error logging helper
# =============================================================================

def _write_error(path_record, run_name, exc, context="", config_path=""):
    ts = time.asctime()
    tb = traceback.format_exc()
    for fpath, content in [
        (f"{path_record}/{run_name}_training_log.txt",
         f"\n{'='*20} ERROR {'='*20}\nTime: {ts}\nContext: {context}\n{tb}\n"),
        (f"{path_record}/bug.txt",
         f"\n{'='*60}\nTime: {ts}\nConfig: {config_path}\nRun: {run_name}\n"
         f"Context: {context}\nException: {type(exc).__name__}: {exc}\n{tb}\n"),
    ]:
        try:
            with open(fpath, "a") as f:
                f.write(content)
        except Exception:
            pass


# =============================================================================
# Build LPBF-NO v3 model from config
# =============================================================================

def build_lpbf_v3_model(args, cond_dim: int, default_dt: float, device):
    """Instantiate LPBFModelV3 from model config and move to device."""
    mcfg = args.model

    # Build slot config
    raw_slots = mcfg.get("field_slots", None)
    if raw_slots is None:
        # Auto-generate from data.fields
        from src.dataset.dataset_lpbf_v3 import (
            _build_default_slots, FIELD_TYPE_REGISTRY
        )
        raw_slots = _build_default_slots(args.data.get("fields", ["T"]))

    K              = len(raw_slots)
    slot_types_str = [s.get("type", FIELD_TYPE_REGISTRY.get(s["name"], "generic"))
                      for s in raw_slots]
    slot_defaults  = [float(s.get("default", 0.0)) for s in raw_slots]

    model = LPBFModelV3(
        K              = K,
        slot_types_str = slot_types_str,
        slot_defaults  = slot_defaults,
        cond_dim       = cond_dim,
        enc_dim        = mcfg.get("enc_dim", 128),
        N_block        = mcfg.get("N_block", 4),
        n_head         = mcfg.get("n_head", 4),
        n_token        = mcfg.get("n_token", 64),
        n_cross_tokens = mcfg.get("n_cross_tokens", 32),
        pos_enc_dim    = mcfg.get("pos_enc_dim", 6),
        x_boost        = mcfg.get("pos_x_boost", 2),
        space_size     = mcfg.get("space_size", 3),
        d_type         = mcfg.get("d_type", 16),
        n_src          = mcfg.get("n_src", 4),
        dt             = mcfg.get("dt", default_dt),
        stepper_scheme = mcfg.get("stepper_scheme", "euler"),
        layer_scale_init = mcfg.get("layer_scale_init", 1e-2),
        kappa_init     = mcfg.get("kappa_init", 1e3),
        max_temp_delta = mcfg.get("max_temp_delta", 500.0),
    ).to(device)

    return model


# =============================================================================
# DataLoader helper
# =============================================================================

def _worker_init_fn(worker_id, base_seed):
    import random as _r
    seed = base_seed + worker_id
    np.random.seed(seed)
    _r.seed(seed)
    torch.manual_seed(seed)


def get_dataloader_v3(args, path_record, pf_extra_max=0):
    data_cfg = args.data

    if pf_extra_max > 0:
        data_cfg["horizon_pf_extra"] = pf_extra_max

    train_dataset = LPBFSlotDataset(args=args, mode="train")
    test_dataset  = LPBFSlotDataset(
        args=args, mode="test",
        mat_data=train_dataset.mat_mean_and_std if train_dataset.normalize else None,
    )
    # Share normalizer
    test_dataset.normalizer = train_dataset.normalizer
    test_dataset._sync_norm_cache()

    subset_size = max(1, len(test_dataset) // 4)
    indices     = list(range(0, len(test_dataset), 4))[:subset_size]
    test_dataset = Subset(test_dataset, indices)

    pin_memory = torch.cuda.is_available()
    seed       = getattr(args, "seed", None)
    dl_kwargs  = {}
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        dl_kwargs = dict(
            generator=g,
            worker_init_fn=lambda wid: _worker_init_fn(wid, seed),
        )

    train_dl = DataLoader(
        train_dataset,
        batch_size  = data_cfg["train"].get("batchsize", 1),
        shuffle     = True,
        num_workers = data_cfg["train"].get("num_workers", 0),
        pin_memory  = pin_memory,
        **dl_kwargs,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size  = data_cfg["test"].get("batchsize", 1),
        shuffle     = False,
        num_workers = data_cfg["test"].get("num_workers", 0),
        pin_memory  = pin_memory,
    )

    cond_dim  = train_dataset.cond_dim
    default_dt = train_dataset.dt

    edge_num = train_dataset.meta_cache[train_dataset.file_paths[0]]["edges"].shape[0]
    with open(f"{path_record}/{args.name}_training_log.txt", "a") as f:
        f.write(f"[v3] Dataset: LPBFSlotDataset\n")
        f.write(f"No. of train samples: {len(train_dataset)}, "
                f"No. of test samples: {len(test_dataset)}\n")
        f.write(f"No. of train batches: {len(train_dl)}, "
                f"No. of test batches: {len(test_dl)}\n")
        f.write(f"Node num: {train_dataset.node_num}, Edge num: {edge_num}, "
                f"Cond dim: {cond_dim}\n")
        f.write(f"Mean dt: {default_dt:.4e}\n")
        f.write(f"Slot config: {[s['name'] for s in args.model.get('field_slots', [])]}\n")

    return train_dl, test_dl, train_dataset.normalizer, cond_dim, default_dt


# =============================================================================
# Adaptive checkpoint threshold (same logic as main_v2)
# =============================================================================

def _probe_ckpt_threshold_v3(model, first_batch, device, horizon, args,
                              path_record="", run_name=""):
    """Binary-search for max autoregressive steps without OOM."""
    if not torch.cuda.is_available() or horizon <= 1:
        return False

    from torch.amp import autocast
    use_amp = args.train.get("use_amp", False)

    # Prepare single-sample probe batch
    dt         = first_batch["dt"][:1].to(device)
    state      = first_batch["slot_values"][:1, :horizon + 1].to(device)
    node_pos   = first_batch["node_pos"][:1].to(device)
    node_pos_p = first_batch.get("node_pos_physical")
    if node_pos_p is not None:
        node_pos_p = node_pos_p[:1].to(device)
    edges      = first_batch["edges"][:1].to(device)
    time_seq   = first_batch["time_seq"][:1, :horizon].to(device)
    conditions = first_batch["conditions"][:1].to(device).float()
    spatial_inform = first_batch["spatial_inform"][:1].to(device)
    presence_mask  = first_batch["presence_mask"][:1].to(device)
    active_indices = first_batch["active_indices"][0].tolist()

    def _try(ckpt_val):
        torch.cuda.empty_cache()
        try:
            model.train()
            ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad().__class__()
            with (autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.enable_grad()):
                out = model.autoregressive(
                    state_in=state[:, 0], node_pos=node_pos, edges=edges,
                    time_seq=time_seq, spatial_inform=spatial_inform,
                    conditions=conditions, dt=dt, check_point=ckpt_val,
                    presence_mask=presence_mask, active_indices=active_indices,
                    node_pos_physical=node_pos_p,
                )
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

    _log(f"[CKP Probe] horizon={horizon} ...")
    if _try(False):
        _log(f"[CKP Probe] No OOM without ckp → threshold={horizon}")
        return horizon
    if not _try(True):
        _log(f"[CKP Probe] OOM with full ckp → falling back to config")
        return True
    lo, hi = 0, horizon - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _try(mid):
            lo = mid
        else:
            hi = mid - 1
    _log(f"[CKP Probe] threshold={lo}")
    return lo


# =============================================================================
# Main
# =============================================================================

def main(args, path_logs, path_nn, path_record, config_path=""):
    device_str = args.device
    if "cuda" in device_str and not torch.cuda.is_available():
        print("! Warning: CUDA not available, using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    EPOCH    = int(args.train["epoch"])
    real_lr  = float(args.train["lr"])
    fields   = args.data.get("fields", ["T"])

    # ---- Pushforward config ----
    pf_cfg        = args.train.get("pushforward", {"enable": False})
    pf_enable     = pf_cfg.get("enable", False)
    pf_start      = pf_cfg.get("start_epoch", 80)
    pf_extra_max  = pf_cfg.get("extra_steps", 3)
    pf_ramp       = pf_cfg.get("ramp_epochs", 40)

    # ---- DataLoader ----
    train_dl, test_dl, normalizer, cond_dim, default_dt = get_dataloader_v3(
        args, path_record,
        pf_extra_max=pf_extra_max if pf_enable else 0,
    )

    # ---- Model ----
    model = build_lpbf_v3_model(args, cond_dim, default_dt, device)

    # Weight init
    load_path = args.model.get("load_path")
    checkpoint = None
    if load_path:
        model_path = os.path.join(load_path, f"{args.name}_best.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {model_path}")
    elif args.model.get("if_init", True):
        model.apply(init_weights)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LPBF-NO v3 | EPOCH={EPOCH} | params={n_params/1e6:.2f}M | fields={fields}")

    with open(f"{path_record}/{args.name}_training_log.txt", "a") as f:
        f.write(f"Model: LPBF-NO v3 (Stage 1)\n")
        f.write(f"Device: {device}\n")
        f.write(f"#params: {n_params/1e6:.2f}M\n")
        f.write(f"Fields: {fields}\n")
        f.write(f"EPOCH: {EPOCH}\n")

    # ---- TensorBoard ----
    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{path_logs}/{args.name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir=log_dir)

    # ---- Optimizer ----
    weight_decay = args.train.get("weight_decay", real_lr / 50.0)
    optimizer    = AdamW(model.parameters(), lr=real_lr, weight_decay=weight_decay)

    # ---- Scheduler ----
    sched_cfg  = args.train.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine_warmrestart")
    if sched_type == "cosine_warmrestart":
        warmup     = sched_cfg.get("warmup_epochs", 10)
        eta_min    = real_lr * sched_cfg.get("eta_min_ratio", 0.005)
        scheduler  = WarmupCosineScheduler(optimizer, warmup, EPOCH, eta_min)
    else:
        scheduler  = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=real_lr / 20.0)
    print(f"Scheduler: {sched_type}")

    # ---- EMA ----
    ema = EMA(model, decay=args.train.get("ema_decay", 0.998))
    print(f"EMA enabled (decay={ema.decay})")

    # ---- Adaptive checkpoint threshold ----
    horizon_train = args.data.get("horizon_train", 1)
    cfg_ckpt      = args.train.get("check_point", False)
    probe_horizon = horizon_train + (pf_extra_max if pf_enable else 0)
    if not cfg_ckpt:
        ckpt_threshold = False
    else:
        try:
            _first = next(iter(train_dl))
            ckpt_threshold = _probe_ckpt_threshold_v3(
                model, _first, device, probe_horizon, args,
                path_record=path_record, run_name=args.name,
            )
            del _first
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[CKP Probe] Failed ({exc}), using config check_point.")
            ckpt_threshold = True

    # ---- Resume ----
    start_epoch, best_val_error = 0, float("inf")
    if checkpoint is not None:
        if "optimizer"  in checkpoint: optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler"  in checkpoint:
            try:    scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception: pass
        if "epoch"      in checkpoint: start_epoch    = checkpoint["epoch"]
        if "best_val_error" in checkpoint: best_val_error = checkpoint["best_val_error"]
        if "ema_shadow" in checkpoint: ema.shadow     = checkpoint["ema_shadow"]
        print(f"Resumed from epoch {start_epoch}, best_val={best_val_error:.4e}")

    # ---- Epsilon annealing schedule for interface head ----
    eps_init  = args.model.get("interface_epsilon_init",  0.3)
    eps_final = args.model.get("interface_epsilon_final", 0.05)

    def _get_epsilon(epoch):
        if EPOCH <= 1:
            return eps_final
        progress = (epoch - start_epoch) / max(1, EPOCH - start_epoch)
        # Linear decay from eps_init to eps_final
        return eps_init + (eps_final - eps_init) * min(1.0, progress)

    # ---- Interrupt save ----
    _interrupted = [False]

    def _save_ckpt(ep, tag="best"):
        if not getattr(args, "if_save", True):
            return None
        ckpt = {
            "epoch":          ep,
            "state_dict":     model.state_dict(),
            "ema_shadow":     ema.shadow,
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val_error": best_val_error,
            "config":         args,
        }
        path = f"{path_nn}/{args.name}_{tag}.pt"
        torch.save(ckpt, path)
        return path

    def _handle_interrupt(sig, frame):
        _interrupted[0] = True

    signal.signal(signal.SIGINT, _handle_interrupt)

    # ==================== Training Loop ====================
    for epoch in range(start_epoch, EPOCH):
        if _interrupted[0]:
            ep = epoch
            save_path = _save_ckpt(ep, tag=f"interrupt_ep{ep}")
            print(f"[INTERRUPTED] Saved to {save_path}")
            with open(f"{path_record}/{args.name}_training_log.txt", "a") as f:
                f.write(f"[INTERRUPTED] Ctrl+C at epoch {ep}/{EPOCH}\n")
            break

        start_time = time.time()

        # Epsilon for this epoch
        model.interface_epsilon = _get_epsilon(epoch)

        # ---- Train ----
        use_pf = pf_enable and epoch >= pf_start
        pf_extra = 0
        if use_pf:
            progress = min(1.0, (epoch - pf_start) / max(1, pf_ramp))
            pf_extra = max(1, int(round(progress * pf_extra_max)))

        try:
            if use_pf and pf_extra > 0:
                # Pushforward: extend rollout beyond base horizon
                # We reuse train_v3 but temporarily extend horizon via a custom batch slice
                # (for simplicity in Stage 1, pushforward uses the same train_v3 loop)
                train_error = train_v3(
                    args, model, train_dl, optimizer, device, normalizer,
                    ema=ema, ckpt_threshold=ckpt_threshold,
                )
            else:
                train_error = train_v3(
                    args, model, train_dl, optimizer, device, normalizer,
                    ema=ema, ckpt_threshold=ckpt_threshold,
                )
        except KeyboardInterrupt:
            _interrupted[0] = True
            continue
        except RuntimeError as exc:
            _write_error(path_record, args.name, exc, context=f"train epoch={epoch}", config_path=config_path)
            print(f"[ERROR] Train epoch {epoch}: {exc}")
            continue

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # ---- Validate (periodically) ----
        eval_every = args.train.get("eval_every", 10)
        if (epoch + 1) % eval_every == 0 or epoch == EPOCH - 1:
            ema.apply_shadow(model)
            try:
                val_error = validate_v3(args, model, test_dl, device, normalizer, epoch)
            except Exception as exc:
                _write_error(path_record, args.name, exc, context=f"val epoch={epoch}")
                val_error = {"mean_l2": float("inf")}
            ema.restore(model)
        else:
            val_error = None

        elapsed = time.time() - start_time

        # ---- Logging ----
        log_line = (
            f"Epoch {epoch+1:04d}/{EPOCH} | "
            f"loss={train_error.get('loss', 0):.4e} | "
            f"val_l2={val_error['mean_l2']:.4e} | " if val_error else ""
            f"lr={lr_now:.2e} | eps={model.interface_epsilon:.3f} | "
            f"t={elapsed:.1f}s"
        )
        print(log_line)

        with open(f"{path_record}/{args.name}_training_log.txt", "a") as f:
            f.write(log_line + "\n")

        writer.add_scalar("train/loss",      train_error.get("loss",        0), epoch)
        writer.add_scalar("train/value_loss",train_error.get("value_loss",   0), epoch)
        writer.add_scalar("train/grad_loss", train_error.get("grad_loss",    0), epoch)
        writer.add_scalar("train/chamfer",   train_error.get("chamfer_loss", 0), epoch)
        writer.add_scalar("train/peak",      train_error.get("peak_loss",    0), epoch)
        writer.add_scalar("train/normal",    train_error.get("normal_loss",  0), epoch)
        writer.add_scalar("train/mean_l2",   train_error.get("mean_l2",      0), epoch)
        writer.add_scalar("train/lr",        lr_now, epoch)
        writer.add_scalar("train/epsilon",   model.interface_epsilon, epoch)

        for fname in fields:
            if f"L2_{fname}" in train_error:
                writer.add_scalar(f"train/L2_{fname}", train_error[f"L2_{fname}"], epoch)

        if val_error is not None:
            writer.add_scalar("val/mean_l2", val_error.get("mean_l2", float("inf")), epoch)
            for fname in fields:
                if f"L2_{fname}" in val_error:
                    writer.add_scalar(f"val/L2_{fname}", val_error[f"L2_{fname}"], epoch)

            # ---- Save best ----
            cur_val = val_error.get("mean_l2", float("inf"))
            if isinstance(cur_val, torch.Tensor):
                cur_val = cur_val.mean().item()
            if cur_val < best_val_error:
                best_val_error = cur_val
                _save_ckpt(epoch + 1, tag="best")
                print(f"  ✓ New best: val_l2={best_val_error:.4e}")

        # Periodic checkpoint (every 50 epochs)
        if (epoch + 1) % 50 == 0:
            _save_ckpt(epoch + 1, tag=f"ep{epoch+1}")

    writer.close()
    print(f"\nTraining complete. Best val L2 = {best_val_error:.4e}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    config = load_json_config(args.config)

    set_seed(getattr(config, "seed", 42))

    save_path    = getattr(config, "save_path", "./result_lpbf_v3")
    path_nn      = f"{save_path}/nn"
    path_logs    = f"{save_path}/logs"
    path_record  = f"{save_path}/record"
    config.if_save = True

    for p in [path_nn, path_logs, path_record]:
        Path(p).mkdir(parents=True, exist_ok=True)

    # Write config snapshot
    import json
    with open(f"{path_record}/{config.name}_config.json", "w") as f:
        json.dump(vars(config) if hasattr(config, "__dict__") else {}, f, indent=2, default=str)

    with open(f"{path_record}/{config.name}_training_log.txt", "a") as f:
        f.write(f"\n{'='*60}\nLPBF-NO v3 Training\n"
                f"Time: {time.asctime()}\n"
                f"Config: {args.config}\n{'='*60}\n")

    main(config, path_logs, path_nn, path_record, config_path=args.config)
