"""
train_lpbf_v3.py — Extended loss functions and training loops for LPBF-NO v3.

Adds on top of src/train.py (never imports from it to avoid side-effects):
  • _compute_chamfer_loss        — soft Chamfer distance on α=0.5 isosurface
  • _compute_peak_loss           — top-k% temperature peak L1 loss
  • _compute_normal_consistency  — interface gradient direction cosine loss
  • get_train_loss_v3            — slot-aware composite loss
  • train_v3                     — training loop for LPBF-NO v3
  • validate_v3                  — validation loop for LPBF-NO v3

All existing src/train.py functions are untouched.
The slot-based model (lpbf_no_v3.Model) is called with the extended
signature:  model.autoregressive(..., presence_mask, active_indices,
                                       node_pos_physical)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.amp import autocast

# Re-use lightweight helpers from original train.py (pure functions, safe to import)
from src.train import (
    _relative_l2,
    _each_l2,
    _rmse,
    _masked_relative_l2,
    _masked_rmse,
    _masked_mse,
    _is_vof_field,
    _build_weight_mask,
    _compute_weighted_value_loss,
    _compute_gradient_loss,
    _init_region_agg,
    _accumulate_region,
    _finalize_region,
    _REGION_PREFIXES,
    _REGION_MEANS,
    compute_spatial_gradient_3d,
)


# =============================================================================
# New loss helpers
# =============================================================================

def _compute_chamfer_loss(
    fld_pred: torch.Tensor,   # [B, T, N, 1]  (normalized, but VoF is [0,1] already)
    fld_gt: torch.Tensor,     # [B, T, N, 1]
    node_pos: torch.Tensor,   # [B, N, 3]  normalized positions
    isovalue: float = 0.5,
    band: float = 0.1,
) -> torch.Tensor:
    """
    Soft Chamfer distance between predicted and GT α=0.5 isosurface nodes.

    Instead of hard thresholding, we use a soft membership weight:
        w(α) = exp(-((α - 0.5) / band)²)

    Then compute the weighted nearest-neighbour distance in physical coords.
    This is fully differentiable and avoids the need for any set operations.

    Complexity:  O(B·T·N) with a random sub-sampling cap of n_max nodes
                 (avoids O(N²) pairwise distance when N is large).

    Args:
        fld_pred, fld_gt: [B, T, N, 1]
        node_pos:         [B, N, 3]
        isovalue:         level-set value (0.5 for VoF)
        band:             Gaussian band width for soft membership
    Returns:
        scalar loss tensor
    """
    B, T, N, _ = fld_pred.shape

    # Soft membership weights: [B, T, N]
    w_pred = torch.exp(-((fld_pred.squeeze(-1) - isovalue) / band) ** 2)  # [B, T, N]
    w_gt   = torch.exp(-((fld_gt.squeeze(-1)   - isovalue) / band) ** 2)

    # Use a sub-sample to cap cost (max 512 "representative" nodes)
    n_max = min(512, N)
    idx   = torch.randperm(N, device=fld_pred.device)[:n_max]

    # Positions: replicate along T, then take subset
    pos_sub = node_pos[:, idx, :]   # [B, n_max, 3]

    # Predicted and GT soft-weighted positions
    wp_sub = w_pred[:, :, idx]      # [B, T, n_max]
    wg_sub = w_gt[:, :, idx]

    # Compute Chamfer per (B, T):
    # For each pred node, find nearest GT node (in soft sense) and vice versa
    pos_exp = pos_sub.unsqueeze(1).expand(B, T, n_max, 3)  # [B, T, n_max, 3]

    # Pairwise squared distances: [B, T, n_max, n_max]
    diff  = pos_exp.unsqueeze(-2) - pos_exp.unsqueeze(-3)   # [B, T, n_max, n_max, 3]
    dist2 = (diff ** 2).sum(-1)                              # [B, T, n_max, n_max]

    # Soft nearest-neighbour distance for pred→gt
    # weight GT nodes by w_gt, take softmin distance
    log_w_gt   = torch.log(wg_sub.unsqueeze(-2) + 1e-8)     # [B, T, 1, n_max]
    log_w_pred = torch.log(wp_sub.unsqueeze(-1) + 1e-8)     # [B, T, n_max, 1]

    soft_min_pred_to_gt = (dist2 * (log_w_gt - dist2).softmax(dim=-1)).sum(-1)   # [B, T, n_max]
    soft_min_gt_to_pred = (dist2 * (log_w_pred - dist2).softmax(dim=-2)).sum(-2) # [B, T, n_max]

    # Weight by isosurface membership
    chamfer_pred = (soft_min_pred_to_gt * wp_sub).sum(-1) / (wp_sub.sum(-1) + 1e-8)
    chamfer_gt   = (soft_min_gt_to_pred * wg_sub).sum(-1) / (wg_sub.sum(-1) + 1e-8)

    return (chamfer_pred + chamfer_gt).mean()


def _compute_peak_loss(
    fld_pred: torch.Tensor,   # [B, T, N, 1]  normalized
    fld_gt: torch.Tensor,     # [B, T, N, 1]  normalized
    top_k_pct: float = 0.01,
) -> torch.Tensor:
    """
    Top-k% peak temperature L1 loss.

    Penalizes under-estimation of peak values — a common failure mode
    when the loss is dominated by low-temperature background nodes.

    Args:
        top_k_pct: fraction of nodes to consider (default 1%)
    Returns:
        scalar loss tensor
    """
    B, T, N, _ = fld_pred.shape
    k = max(1, int(N * top_k_pct))

    pred_flat = fld_pred.reshape(B * T, N)     # [B*T, N]
    gt_flat   = fld_gt.reshape(B * T, N)

    # Top-k indices by GT value (find the hottest nodes)
    _, top_idx = gt_flat.topk(k, dim=-1)       # [B*T, k]
    pred_top   = pred_flat.gather(1, top_idx)  # [B*T, k]
    gt_top     = gt_flat.gather(1, top_idx)

    return F.l1_loss(pred_top, gt_top)


def _compute_normal_consistency(
    fld_pred: torch.Tensor,   # [B, T, N, 1]
    fld_gt: torch.Tensor,     # [B, T, N, 1]
    grid_shape,               # as in compute_spatial_gradient_3d
) -> torch.Tensor:
    """
    Interface normal consistency:  1 - mean(cos(∇α_pred, ∇α_gt))

    Enforces that predicted interface normals align with GT normals,
    independently of the magnitude — ensures correct interface orientation.
    """
    pred_gx, pred_gy, pred_gz = compute_spatial_gradient_3d(fld_pred, grid_shape)
    gt_gx,   gt_gy,   gt_gz   = compute_spatial_gradient_3d(fld_gt,   grid_shape)

    def _align_shapes(a, b):
        """Trim to the smaller shape."""
        for dim in range(a.dim()):
            sz = min(a.shape[dim], b.shape[dim])
            a = a.narrow(dim, 0, sz)
            b = b.narrow(dim, 0, sz)
        return a, b

    pred_gx, gt_gx = _align_shapes(pred_gx, gt_gx)
    pred_gy, gt_gy = _align_shapes(pred_gy, gt_gy)
    pred_gz, gt_gz = _align_shapes(pred_gz, gt_gz)

    pred_norm_sq = pred_gx ** 2 + pred_gy ** 2 + pred_gz ** 2
    gt_norm_sq   = gt_gx   ** 2 + gt_gy   ** 2 + gt_gz   ** 2

    dot = pred_gx * gt_gx + pred_gy * gt_gy + pred_gz * gt_gz
    cos = dot / (torch.sqrt(pred_norm_sq * gt_norm_sq) + 1e-8)

    return (1.0 - cos).mean()


# =============================================================================
# Slot-aware composite loss (get_train_loss_v3)
# =============================================================================

def get_train_loss_v3(
    slot_names: list,          # [K] slot names (str)
    slot_types_str: list,      # [K] slot types ('temperature'|'interface'|...)
    active_indices: list,      # indices of active slots in this batch
    predict_hat: torch.Tensor, # [B, T, N, K]
    label_gt: torch.Tensor,    # [B, T, N, K]
    normalizer,                # ChannelNormalizer  (for active field channels)
    weight_cfg: dict,
    node_pos: torch.Tensor,    # [B, N, 3]
    active_mask=None,
) -> dict:
    """
    Slot-aware training loss for LPBF-NO v3.

    Only computes loss for active_indices slots; inactive slots are masked out.
    Extends get_train_loss with:
      - Chamfer loss for interface slots (weight_cfg["chamfer"])
      - Peak temperature loss for temperature slots (weight_cfg["peak"])
      - Normal consistency for interface slots (weight_cfg["normal"])

    normalizer is expected to cover only the active fields in the order
    they appear in active_indices (same as LPBFSlotDataset.normalizer).
    """
    n_active = float(len(active_indices))

    losses = {
        "value_loss":  torch.tensor(0.0, device=predict_hat.device),
        "grad_loss":   torch.tensor(0.0, device=predict_hat.device),
        "chamfer_loss":torch.tensor(0.0, device=predict_hat.device),
        "peak_loss":   torch.tensor(0.0, device=predict_hat.device),
        "normal_loss": torch.tensor(0.0, device=predict_hat.device),
        "loss":        0.0,
        "mean_l2":     0.0,
    }

    pred_fp32  = predict_hat.float()
    label_fp32 = label_gt.float()

    # Build per-active-slot normalizer index map
    # normalizer covers active fields in order of active_indices
    with torch.no_grad():
        # Denormalize only active channels
        pred_real_list  = []
        label_real_list = []
        for local_i, si in enumerate(active_indices):
            mean_i = normalizer.mean[..., local_i].to(predict_hat.device)
            std_i  = normalizer.std[..., local_i].to(predict_hat.device)
            eps_i  = normalizer.eps
            pred_real_list.append(pred_fp32[..., si:si+1] * (std_i + eps_i) + mean_i)
            label_real_list.append(label_fp32[..., si:si+1] * (std_i + eps_i) + mean_i)

    weight_masks = {}

    # ---- Value loss ----
    use_weighted = weight_cfg.get("enable", False)
    weight_field = weight_cfg.get("field", [])
    thresholds   = weight_cfg.get("threshold", [])
    bws          = weight_cfg.get("base_weight", [])
    fws          = weight_cfg.get("focus_weight", [])

    error_list = []
    for local_i, si in enumerate(active_indices):
        fname    = slot_names[si]
        fld_pred = pred_fp32[..., si:si+1]
        fld_gt   = label_fp32[..., si:si+1]

        if use_weighted and fname in weight_field:
            idx       = weight_field.index(fname)
            thresh    = thresholds[idx]
            bw, fw    = float(bws[idx]), float(fws[idx])
            fld_gt_r  = label_real_list[local_i]
            is_vof    = _is_vof_field(fname)
            loss_t, w_mask = _compute_weighted_value_loss(fld_pred, fld_gt, fld_gt_r, thresh, bw, fw, is_vof)
            error_list.append(loss_t)
            weight_masks[fname] = w_mask
        else:
            error_list.append((fld_pred - fld_gt) ** 2)

    if error_list:
        losses["value_loss"] = torch.stack([e.mean() for e in error_list]).mean()

    # ---- Gradient loss ----
    if weight_cfg.get("gradient", False):
        grad_weights = weight_cfg.get("grad_weight", {})
        grid_shape   = weight_cfg.get("grid_shape", None)
        if grid_shape is not None:
            grad_total = torch.tensor(0.0, device=predict_hat.device)
            for local_i, si in enumerate(active_indices):
                fname = slot_names[si]
                if fname in grad_weights:
                    gw     = float(grad_weights[fname])
                    is_vof = _is_vof_field(fname)
                    loss_g = _compute_gradient_loss(
                        pred_fp32[..., si:si+1],
                        label_fp32[..., si:si+1],
                        grid_shape, gw, is_vof, weight_cfg,
                        weight_mask=weight_masks.get(fname)
                    )
                    grad_total = grad_total + loss_g
            losses["grad_loss"] = grad_total

    # ---- Chamfer loss (interface slots) ----
    if weight_cfg.get("chamfer", False):
        chamfer_w = float(weight_cfg.get("chamfer_weight", 0.1))
        chamfer_total = torch.tensor(0.0, device=predict_hat.device)
        for local_i, si in enumerate(active_indices):
            if slot_types_str[si] == "interface":
                ch_loss = _compute_chamfer_loss(
                    pred_fp32[..., si:si+1],
                    label_fp32[..., si:si+1],
                    node_pos,
                )
                chamfer_total = chamfer_total + ch_loss
        losses["chamfer_loss"] = chamfer_total * chamfer_w

    # ---- Peak temperature loss ----
    if weight_cfg.get("peak", False):
        peak_w = float(weight_cfg.get("peak_weight", 0.5))
        peak_total = torch.tensor(0.0, device=predict_hat.device)
        for local_i, si in enumerate(active_indices):
            if slot_types_str[si] == "temperature":
                pk_loss = _compute_peak_loss(
                    pred_fp32[..., si:si+1],
                    label_fp32[..., si:si+1],
                    top_k_pct=weight_cfg.get("peak_top_k_pct", 0.01),
                )
                peak_total = peak_total + pk_loss
        losses["peak_loss"] = peak_total * peak_w

    # ---- Normal consistency loss (interface slots) ----
    if weight_cfg.get("normal", False):
        normal_w   = float(weight_cfg.get("normal_weight", 0.1))
        grid_shape = weight_cfg.get("grid_shape", None)
        if grid_shape is not None:
            normal_total = torch.tensor(0.0, device=predict_hat.device)
            for local_i, si in enumerate(active_indices):
                if slot_types_str[si] == "interface":
                    n_loss = _compute_normal_consistency(
                        pred_fp32[..., si:si+1],
                        label_fp32[..., si:si+1],
                        grid_shape,
                    )
                    normal_total = normal_total + n_loss
            losses["normal_loss"] = normal_total * normal_w

    # ---- Total loss ----
    grad_mult   = float(weight_cfg.get("grad_loss_multiplier", 8.0))
    losses["loss"] = (
        losses["value_loss"]
        + grad_mult * losses["grad_loss"]
        + losses["chamfer_loss"]
        + losses["peak_loss"]
        + losses["normal_loss"]
    )

    # ---- Monitoring metrics ----
    with torch.no_grad():
        for local_i, si in enumerate(active_indices):
            fname       = slot_names[si]
            pr          = pred_real_list[local_i]
            lr          = label_real_list[local_i]
            rel_l2      = _relative_l2(pr, lr)
            losses[f"L2_{fname}"]   = rel_l2
            losses["mean_l2"]       = losses["mean_l2"] + rel_l2 / n_active
            losses[f"RMSE_{fname}"] = _rmse(pr, lr).item() if pr.ndim == 4 else 0.0

        losses["each_l2"] = _each_l2(
            torch.stack([pred_real_list[i] for i in range(len(active_indices))], dim=-1).squeeze(-2)
            if len(active_indices) == 1 else
            torch.cat(pred_real_list, dim=-1),
            torch.stack([label_real_list[i] for i in range(len(active_indices))], dim=-1).squeeze(-2)
            if len(active_indices) == 1 else
            torch.cat(label_real_list, dim=-1),
        )

    return losses


# =============================================================================
# Training loop for LPBF-NO v3
# =============================================================================

def train_v3(
    args,
    model,
    train_dataloader,
    optim,
    device,
    normalizer,
    ema=None,
    ckpt_threshold=None,
):
    """
    Training loop for LPBF-NO v3.

    Differences from train_v2:
      - Reads slot_values / presence_mask / active_indices from batch
      - Passes node_pos_physical to model.autoregressive
      - Calls get_train_loss_v3 instead of get_train_loss
    """
    data_cfg    = args.data
    train_cfg   = args.train
    model_cfg   = args.model

    horizon           = data_cfg.get("horizon_train", 1)
    fields_active     = data_cfg.get("fields", ["T"])   # active field names
    use_amp           = train_cfg.get("use_amp", False)
    check_point       = ckpt_threshold if ckpt_threshold is not None else train_cfg.get("check_point", False)
    weight_loss       = train_cfg.get("weight_loss", {"enable": False})
    grad_loss_weight  = train_cfg.get("grad_loss_weight", 8.0)

    # Slot metadata
    raw_slots     = model_cfg.get("field_slots", [{"name": f, "type": "generic"} for f in fields_active])
    slot_names    = [s["name"] for s in raw_slots]
    slot_types_str = [s.get("type", "generic") for s in raw_slots]

    agg = {
        "loss": 0.0, "value_loss": 0.0, "grad_loss": 0.0,
        "chamfer_loss": 0.0, "peak_loss": 0.0, "normal_loss": 0.0,
        "mean_l2": 0.0,
        "each_l2": torch.zeros(horizon, device=device),
        "num": 0,
    }
    for fname in fields_active:
        agg[f"L2_{fname}"] = 0.0
        agg[f"RMSE_{fname}"] = 0.0

    model.train()
    normalizer.to(device)

    pbar = tqdm(train_dataloader, desc="  Train-v3", unit="bt", leave=True, ncols=120, colour="green")
    for batch in pbar:
        dt         = batch["dt"].to(device)
        state      = batch["slot_values"][:, :horizon + 1].to(device)    # [B, T+1, N, K]
        node_pos   = batch["node_pos"].to(device)
        node_pos_p = batch.get("node_pos_physical")
        if node_pos_p is not None:
            node_pos_p = node_pos_p.to(device)
        edges      = batch["edges"].to(device)
        time_seq   = batch["time_seq"][:, :horizon].to(device)
        conditions = batch["conditions"].to(device).float()
        spatial_inform = batch["spatial_inform"].to(device)
        presence_mask  = batch["presence_mask"].to(device)     # [B, K]
        # active_indices are the same for all samples in a batch
        active_indices = batch["active_indices"][0].tolist()   # take from first sample

        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch["grid_shape"].numpy()

        batch_num = state.shape[0]

        def _rollout():
            return model.autoregressive(
                state_in          = state[:, 0],
                node_pos          = node_pos,
                edges             = edges,
                time_seq          = time_seq,
                spatial_inform    = spatial_inform,
                conditions        = conditions,
                dt                = dt,
                check_point       = check_point,
                presence_mask     = presence_mask,
                active_indices    = active_indices,
                node_pos_physical = node_pos_p,
            )   # [B, T, N, K]

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                predict_hat = _rollout()
                costs = get_train_loss_v3(
                    slot_names, slot_types_str, active_indices,
                    predict_hat, state[:, 1:],
                    normalizer, weight_loss, node_pos,
                )
        else:
            predict_hat = _rollout()
            costs = get_train_loss_v3(
                slot_names, slot_types_str, active_indices,
                predict_hat, state[:, 1:],
                normalizer, weight_loss, node_pos,
            )

        loss = costs["loss"]

        # NaN / spike guard
        _skip = False
        if not torch.isfinite(loss):
            _skip = True
        else:
            cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
            if cur_avg is not None and loss.item() > 10 * cur_avg:
                _skip = True
        if _skip:
            optim.zero_grad()
            del predict_hat, costs, loss
            torch.cuda.empty_cache()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.get("grad_clip", 1.0))
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            optim.zero_grad()
            del predict_hat, costs, loss
            torch.cuda.empty_cache()
            continue
        optim.step()
        optim.zero_grad()
        if ema is not None:
            ema.update(model)

        # Accumulate
        agg["loss"]          += costs["loss"].item() * batch_num
        agg["value_loss"]    += costs["value_loss"].item() * batch_num
        agg["grad_loss"]     += costs["grad_loss"].item() * batch_num
        agg["chamfer_loss"]  += costs["chamfer_loss"].item() * batch_num
        agg["peak_loss"]     += costs["peak_loss"].item() * batch_num
        agg["normal_loss"]   += costs["normal_loss"].item() * batch_num
        agg["mean_l2"]       += costs["mean_l2"].mean().item() * batch_num
        agg["each_l2"]       += costs["each_l2"] * batch_num
        agg["num"]           += batch_num

        for fname in fields_active:
            if f"L2_{fname}" in costs:
                agg[f"L2_{fname}"]   += costs[f"L2_{fname}"].mean().item() * batch_num
            if f"RMSE_{fname}" in costs:
                agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num

        pbar.set_postfix({"Loss": f"{agg['loss']/agg['num']:.4e}"})

    n = agg.pop("num")
    for k in list(agg.keys()):
        if k != "each_l2":
            agg[k] = agg[k] / n if n > 0 else 0.0
    agg["each_l2"] = (agg["each_l2"] / n).cpu()
    return agg


def validate_v3(
    args,
    model,
    val_dataloader,
    device,
    normalizer,
    epoch: int,
):
    """Validation loop for LPBF-NO v3."""
    data_cfg   = args.data
    train_cfg  = args.train
    model_cfg  = args.model

    horizon        = data_cfg.get("horizon_test", 1)
    fields_active  = data_cfg.get("fields", ["T"])
    use_amp        = train_cfg.get("use_amp", False)
    check_point    = train_cfg.get("check_point", False)

    raw_slots      = model_cfg.get("field_slots", [{"name": f, "type": "generic"} for f in fields_active])
    slot_names     = [s["name"] for s in raw_slots]
    slot_types_str = [s.get("type", "generic") for s in raw_slots]

    agg = {
        "mean_l2": 0.0,
        "each_l2": torch.zeros(horizon, device=device),
        "num": 0,
    }
    for fname in fields_active:
        agg[f"L2_{fname}"]   = 0.0
        agg[f"RMSE_{fname}"] = 0.0

    model.eval()
    normalizer.to(device)

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="  Valid-v3", unit="bt", leave=False, ncols=120, colour="yellow")
        for batch in pbar:
            state      = batch["slot_values"].to(device)
            node_pos   = batch["node_pos"].to(device)
            node_pos_p = batch.get("node_pos_physical")
            if node_pos_p is not None:
                node_pos_p = node_pos_p.to(device)
            edges      = batch["edges"].to(device)
            time_seq   = batch["time_seq"].to(device)
            conditions = batch["conditions"].to(device).float()
            spatial_inform = batch["spatial_inform"].to(device)
            presence_mask  = batch["presence_mask"].to(device)
            active_indices = batch["active_indices"][0].tolist()
            dt             = batch["dt"].to(device)

            batch_num = state.shape[0]

            if use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    predict_hat = model.autoregressive(
                        state_in=state[:, 0], node_pos=node_pos, edges=edges,
                        time_seq=time_seq, spatial_inform=spatial_inform,
                        conditions=conditions, dt=dt, check_point=check_point,
                        presence_mask=presence_mask, active_indices=active_indices,
                        node_pos_physical=node_pos_p,
                    )
            else:
                predict_hat = model.autoregressive(
                    state_in=state[:, 0], node_pos=node_pos, edges=edges,
                    time_seq=time_seq, spatial_inform=spatial_inform,
                    conditions=conditions, dt=dt, check_point=check_point,
                    presence_mask=presence_mask, active_indices=active_indices,
                    node_pos_physical=node_pos_p,
                )

            # Compute validation metrics (value only, no grad)
            pred_fp32  = predict_hat.float()
            label_fp32 = state[:, 1:].float()

            for local_i, si in enumerate(active_indices):
                fname = slot_names[si]
                mean_i = normalizer.mean[..., local_i].to(device)
                std_i  = normalizer.std[..., local_i].to(device)
                eps_i  = normalizer.eps
                pr_r   = pred_fp32[..., si:si+1]  * (std_i + eps_i) + mean_i
                gt_r   = label_fp32[..., si:si+1] * (std_i + eps_i) + mean_i

                rel_l2 = _relative_l2(pr_r, gt_r)
                if f"L2_{fname}" in agg:
                    agg[f"L2_{fname}"]   += rel_l2.mean().item() * batch_num
                if f"RMSE_{fname}" in agg:
                    agg[f"RMSE_{fname}"] += _rmse(pr_r, gt_r).mean().item() * batch_num
                agg["mean_l2"] += rel_l2.mean().item() / len(active_indices) * batch_num

            agg["num"] += batch_num

    n = agg.pop("num")
    for k in list(agg.keys()):
        if k != "each_l2":
            agg[k] = agg[k] / n if n > 0 else 0.0
    agg["each_l2"] = (agg["each_l2"] / n).cpu()
    return agg
