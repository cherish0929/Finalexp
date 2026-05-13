"""Training and validation loss functions."""

import math
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from .helpers import _is_vof_field
from .gradient import compute_spatial_gradient_3d
from .metrics import _relative_l2, _each_l2, _rmse, _masked_relative_l2, _masked_rmse, _masked_mse


# ---------------------------------------------------------------------------
# Weight mask construction
# ---------------------------------------------------------------------------

def _build_weight_mask(fld_gt_real, thresh, bw, fw):
    """
    Build spatial weight mask from ground-truth values.
    Args:
        fld_gt_real: [B, T, N, 1] denormalized ground truth
        thresh: float or [low, high] list
        bw, fw: float, background/focus weights
    Returns:
        weight_mask: [B, T, N, 1], mean-normalized weights
    """
    with torch.no_grad():
        if isinstance(thresh, list):
            condition = (fld_gt_real > thresh[0]) & (fld_gt_real < thresh[1])
            weight_mask = torch.where(condition, fw, bw)
        else:
            weight_mask = torch.where(fld_gt_real > thresh, fw, bw)
        weight_mask = weight_mask / torch.mean(weight_mask)
    return weight_mask


# ---------------------------------------------------------------------------
# Single-field value loss
# ---------------------------------------------------------------------------

def _compute_weighted_value_loss(fld_pred, fld_gt, fld_gt_real, thresh, bw, fw, is_vof):
    """
    Compute weighted value loss for a single field.
    Args:
        fld_pred, fld_gt: [B, T, N, 1] normalized
        fld_gt_real: [B, T, N, 1] denormalized (for weight mask)
        thresh: float or [low, high]
        bw, fw: float
        is_vof: bool
    Returns:
        (loss_tensor, weight_mask)
    """
    weight_mask = _build_weight_mask(fld_gt_real, thresh, bw, fw)

    if is_vof:
        error = fld_pred - fld_gt
        abs_error = torch.abs(error)
        with torch.no_grad():
            delta = torch.clamp(torch.median(abs_error), min=1e-3)
        huber = torch.where(
            abs_error <= delta,
            0.5 * error ** 2,
            delta * (abs_error - 0.5 * delta)
        )
        loss_tensor = huber * weight_mask
    else:
        squared_error = (fld_pred - fld_gt) ** 2
        loss_tensor = squared_error * weight_mask

    return loss_tensor, weight_mask


# ---------------------------------------------------------------------------
# Gradient loss helpers
# ---------------------------------------------------------------------------

def _adapt_mask_to_gradient(weight_mask, grid_shape, axis):
    """
    Adapt spatial weight_mask from [B, T, N, 1] to gradient-differenced shape.
    Takes the max of adjacent elements along the differenced axis.
    """
    B, T, N, C = weight_mask.shape
    Nx, Ny, Nz = int(grid_shape[0][0]), int(grid_shape[0][1]), int(grid_shape[0][2])
    grid_mask = weight_mask.view(B, T, Nz, Ny, Nx, C)

    if axis == 'x':
        return torch.max(grid_mask[:, :, :, :, 1:, :], grid_mask[:, :, :, :, :-1, :])
    elif axis == 'y':
        return torch.max(grid_mask[:, :, :, 1:, :, :], grid_mask[:, :, :, :-1, :, :])
    elif axis == 'z':
        return torch.max(grid_mask[:, :, 1:, :, :, :], grid_mask[:, :, :-1, :, :, :])
    else:
        raise ValueError(f"Unknown axis: {axis}")


def _compute_gradient_loss(fld_pred, fld_gt, grid_shape, gw, is_vof, weight_cfg, weight_mask=None):
    """
    Compute spatial gradient loss for a single field.
    Args:
        fld_pred, fld_gt: [B, T, N, 1] normalized
        grid_shape: numpy array
        gw: gradient weight coefficient for this field
        is_vof: bool
        weight_cfg: full weight_loss config dict
        weight_mask: optional [B, T, N, 1] spatial weight from value loss
    Returns:
        scalar loss tensor
    """
    pred_gx, pred_gy, pred_gz = compute_spatial_gradient_3d(fld_pred, grid_shape)
    gt_gx, gt_gy, gt_gz = compute_spatial_gradient_3d(fld_gt, grid_shape)

    if is_vof:
        mode = weight_cfg.get("grad_weight_mode", "mask")

        if mode == "adaptive":
            loss_g = 0
            for pred_g, gt_g in [(pred_gx, gt_gx), (pred_gy, gt_gy), (pred_gz, gt_gz)]:
                with torch.no_grad():
                    gt_mag = torch.abs(gt_g)
                    w = 1.0 + gt_mag / (torch.mean(gt_mag) + 1e-8)
                    w = w / torch.mean(w)
                loss_g = loss_g + torch.mean(w * (pred_g - gt_g) ** 2)
        else:  # mode == "mask"
            if weight_mask is not None:
                wmx = _adapt_mask_to_gradient(weight_mask, grid_shape, 'x')
                wmy = _adapt_mask_to_gradient(weight_mask, grid_shape, 'y')
                wmz = _adapt_mask_to_gradient(weight_mask, grid_shape, 'z')
                loss_g = (torch.mean(wmx * (pred_gx - gt_gx) ** 2)
                          + torch.mean(wmy * (pred_gy - gt_gy) ** 2)
                          + torch.mean(wmz * (pred_gz - gt_gz) ** 2))
            else:
                loss_g = (F.mse_loss(pred_gx, gt_gx)
                          + F.mse_loss(pred_gy, gt_gy)
                          + F.mse_loss(pred_gz, gt_gz))
    else:
        loss_g = (F.mse_loss(pred_gx, gt_gx)
                  + F.mse_loss(pred_gy, gt_gy)
                  + F.mse_loss(pred_gz, gt_gz))

    return loss_g * gw


# ---------------------------------------------------------------------------
# LPBF-specific losses: Chamfer, Peak Temperature, Normal Consistency
# ---------------------------------------------------------------------------

def _compute_chamfer_loss(
    fld_pred: torch.Tensor,
    fld_gt: torch.Tensor,
    node_pos: torch.Tensor,
    isovalue: float = 0.5,
    band: float = 0.1,
) -> torch.Tensor:
    """
    Soft Chamfer distance between predicted and GT isosurface nodes.
    Uses Gaussian membership weight and sub-sampled pairwise distance.
    """
    B, T, N, _ = fld_pred.shape

    w_pred = torch.exp(-((fld_pred.squeeze(-1) - isovalue) / band) ** 2)
    w_gt = torch.exp(-((fld_gt.squeeze(-1) - isovalue) / band) ** 2)

    n_max = min(512, N)
    idx = torch.randperm(N, device=fld_pred.device)[:n_max]

    pos_sub = node_pos[:, idx, :]
    wp_sub = w_pred[:, :, idx]
    wg_sub = w_gt[:, :, idx]

    pos_exp = pos_sub.unsqueeze(1).expand(B, T, n_max, 3)
    diff = pos_exp.unsqueeze(-2) - pos_exp.unsqueeze(-3)
    dist2 = (diff ** 2).sum(-1)

    log_w_gt = torch.log(wg_sub.unsqueeze(-2) + 1e-8)
    log_w_pred = torch.log(wp_sub.unsqueeze(-1) + 1e-8)

    soft_min_pred_to_gt = (dist2 * (log_w_gt - dist2).softmax(dim=-1)).sum(-1)
    soft_min_gt_to_pred = (dist2 * (log_w_pred - dist2).softmax(dim=-2)).sum(-2)

    chamfer_pred = (soft_min_pred_to_gt * wp_sub).sum(-1) / (wp_sub.sum(-1) + 1e-8)
    chamfer_gt = (soft_min_gt_to_pred * wg_sub).sum(-1) / (wg_sub.sum(-1) + 1e-8)

    return (chamfer_pred + chamfer_gt).mean()


def _compute_peak_loss(
    fld_pred: torch.Tensor,
    fld_gt: torch.Tensor,
    top_k_pct: float = 0.01,
) -> torch.Tensor:
    """Top-k% peak value L1 loss. Penalizes under-estimation of peak values."""
    B, T, N, _ = fld_pred.shape
    k = max(1, int(N * top_k_pct))

    pred_flat = fld_pred.reshape(B * T, N)
    gt_flat = fld_gt.reshape(B * T, N)

    _, top_idx = gt_flat.topk(k, dim=-1)
    pred_top = pred_flat.gather(1, top_idx)
    gt_top = gt_flat.gather(1, top_idx)

    return F.l1_loss(pred_top, gt_top)


def _compute_normal_consistency(
    fld_pred: torch.Tensor,
    fld_gt: torch.Tensor,
    grid_shape,
) -> torch.Tensor:
    """Interface normal consistency: 1 - mean(cos(∇α_pred, ∇α_gt))."""
    pred_gx, pred_gy, pred_gz = compute_spatial_gradient_3d(fld_pred, grid_shape)
    gt_gx, gt_gy, gt_gz = compute_spatial_gradient_3d(fld_gt, grid_shape)

    # Gradients have different shapes along spatial dims (finite diff reduces by 1
    # along the differenced axis). Trim all to the common minimum shape.
    min_z = min(pred_gx.shape[2], pred_gy.shape[2], pred_gz.shape[2])
    min_y = min(pred_gx.shape[3], pred_gy.shape[3], pred_gz.shape[3])
    min_x = min(pred_gx.shape[4], pred_gy.shape[4], pred_gz.shape[4])

    def _trim(t):
        return t[:, :, :min_z, :min_y, :min_x, :]

    pred_gx, pred_gy, pred_gz = _trim(pred_gx), _trim(pred_gy), _trim(pred_gz)
    gt_gx, gt_gy, gt_gz = _trim(gt_gx), _trim(gt_gy), _trim(gt_gz)

    pred_norm_sq = pred_gx ** 2 + pred_gy ** 2 + pred_gz ** 2
    gt_norm_sq = gt_gx ** 2 + gt_gy ** 2 + gt_gz ** 2

    dot = pred_gx * gt_gx + pred_gy * gt_gy + pred_gz * gt_gz
    cos = dot / (torch.sqrt(pred_norm_sq * gt_norm_sq + 1e-8) + 1e-8)

    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Reserved loss stubs (not yet implemented)
# ---------------------------------------------------------------------------

def _compute_laplacian_loss(fld_pred, fld_gt, grid_shape, lw):
    """Second-order spatial gradient (Laplacian) loss — not yet implemented."""
    raise NotImplementedError("Laplacian loss is not yet implemented. Set laplacian: false in config.")


def _compute_sharpness_loss(fld_pred, fld_gt_real, thresh, sw):
    """Interface sharpness loss — not yet implemented."""
    raise NotImplementedError("Sharpness loss is not yet implemented. Set sharpness: false in config.")


# ---------------------------------------------------------------------------
# Composite training loss
# ---------------------------------------------------------------------------

def get_train_loss(fields, predict_hat, label_gt, normalizer, weight_cfg: dict, active_mask=None, node_pos=None, model=None):
    """Return loss tensor and monitoring metrics (floats)."""
    num_channels = float(len(fields))

    losses = {
        "value_loss": torch.tensor(0),
        "grad_loss": torch.tensor(0),
        "loss": 0,
        'mean_l2': 0,
    }

    pred_fp32 = predict_hat.float()
    label_fp32 = label_gt.float()
    with torch.no_grad():
        pred_real = normalizer.denormalize(pred_fp32)
        label_real = normalizer.denormalize(label_fp32)

    # ---- value loss ----
    weight_masks = {}

    if weight_cfg.get("enable", False):
        error_list = []
        weight_field = weight_cfg.get("field", [])
        thresholds = weight_cfg.get("threshold", [])
        bws = weight_cfg.get("base_weight", [])
        fws = weight_cfg.get("focus_weight", [])

        for idx, fld in enumerate(fields):
            fld_pred = pred_fp32[..., idx:idx + 1]
            fld_gt = label_fp32[..., idx:idx + 1]

            if fld in weight_field:
                i = weight_field.index(fld)
                thresh = thresholds[i]
                bw, fw = float(bws[i]), float(fws[i])
                fld_gt_real = label_real[..., idx:idx + 1]
                is_vof = _is_vof_field(fld)

                loss_tensor, w_mask = _compute_weighted_value_loss(
                    fld_pred, fld_gt, fld_gt_real, thresh, bw, fw, is_vof
                )
                error_list.append(loss_tensor)
                weight_masks[fld] = w_mask
            else:
                error_list.append((fld_pred - fld_gt) ** 2)

        error_map = torch.cat(error_list, dim=-1)
        losses["value_loss"] = torch.mean(error_map)
    else:
        losses["value_loss"] = F.mse_loss(pred_fp32, label_fp32)

    # ---- gradient loss ----
    if weight_cfg.get("gradient", False):
        grad_loss_total = 0
        grad_weights = weight_cfg.get("grad_weight", {})
        grid_shape = weight_cfg.get("grid_shape", None)

        for idx, fld in enumerate(fields):
            if fld in grad_weights:
                gw = float(grad_weights[fld])
                is_vof = _is_vof_field(fld)
                fld_pred_g = pred_fp32[..., idx:idx + 1]
                fld_gt_g = label_fp32[..., idx:idx + 1]

                loss_g = _compute_gradient_loss(
                    fld_pred_g, fld_gt_g, grid_shape, gw,
                    is_vof, weight_cfg, weight_mask=weight_masks.get(fld)
                )
                grad_loss_total = grad_loss_total + loss_g

        losses["grad_loss"] = grad_loss_total

    # ---- Reserved: laplacian / sharpness loss ----
    if weight_cfg.get("laplacian", False):
        lap_weights = weight_cfg.get("laplacian_weight", {})
        grid_shape = weight_cfg.get("grid_shape", None)
        lap_total = 0
        for idx, fld in enumerate(fields):
            if _is_vof_field(fld) and fld in lap_weights:
                lw = float(lap_weights[fld])
                lap_total = lap_total + _compute_laplacian_loss(
                    pred_fp32[..., idx:idx + 1], label_fp32[..., idx:idx + 1],
                    grid_shape, lw
                )
        losses["laplacian_loss"] = lap_total

    if weight_cfg.get("sharpness", False):
        sharp_weights = weight_cfg.get("sharpness_weight", {})
        thresholds = weight_cfg.get("threshold", [])
        weight_field = weight_cfg.get("field", [])
        sharp_total = 0
        for idx, fld in enumerate(fields):
            if _is_vof_field(fld) and fld in sharp_weights:
                sw = float(sharp_weights[fld])
                i = weight_field.index(fld) if fld in weight_field else None
                thresh = thresholds[i] if i is not None else [0.1, 0.9]
                sharp_total = sharp_total + _compute_sharpness_loss(
                    pred_fp32[..., idx:idx + 1], label_real[..., idx:idx + 1],
                    thresh, sw
                )
        losses["sharpness_loss"] = sharp_total

    # ---- LPBF-specific losses: chamfer, peak, normal ----
    if weight_cfg.get("chamfer", False) and node_pos is not None:
        chamfer_w = float(weight_cfg.get("chamfer_weight", 0.1))
        chamfer_total = torch.tensor(0.0, device=predict_hat.device)
        for idx, fld in enumerate(fields):
            if _is_vof_field(fld):
                chamfer_total = chamfer_total + _compute_chamfer_loss(
                    pred_fp32[..., idx:idx + 1],
                    label_fp32[..., idx:idx + 1],
                    node_pos,
                )
        losses["chamfer_loss"] = chamfer_total * chamfer_w

    if weight_cfg.get("peak", False):
        peak_w = float(weight_cfg.get("peak_weight", 0.5))
        peak_total = torch.tensor(0.0, device=predict_hat.device)
        for idx, fld in enumerate(fields):
            if fld == "T":
                peak_total = peak_total + _compute_peak_loss(
                    pred_fp32[..., idx:idx + 1],
                    label_fp32[..., idx:idx + 1],
                    top_k_pct=weight_cfg.get("peak_top_k_pct", 0.01),
                )
        losses["peak_loss"] = peak_total * peak_w

    if weight_cfg.get("normal", False) and node_pos is not None:
        normal_w = float(weight_cfg.get("normal_weight", 0.1))
        grid_shape_n = weight_cfg.get("grid_shape", None)
        if grid_shape_n is not None:
            normal_total = torch.tensor(0.0, device=predict_hat.device)
            for idx, fld in enumerate(fields):
                if _is_vof_field(fld):
                    normal_total = normal_total + _compute_normal_consistency(
                        pred_fp32[..., idx:idx + 1],
                        label_fp32[..., idx:idx + 1],
                        grid_shape_n,
                    )
            losses["normal_loss"] = normal_total * normal_w

    # ---- Total loss ----
    grad_multiplier = float(weight_cfg.get("grad_loss_multiplier", 8.0))
    losses["loss"] = losses["value_loss"] + grad_multiplier * losses["grad_loss"]

    if "laplacian_loss" in losses:
        lap_multiplier = float(weight_cfg.get("laplacian_loss_multiplier", 0.05))
        losses["loss"] = losses["loss"] + lap_multiplier * losses["laplacian_loss"]
    if "sharpness_loss" in losses:
        sharp_multiplier = float(weight_cfg.get("sharpness_loss_multiplier", 0.2))
        losses["loss"] = losses["loss"] + sharp_multiplier * losses["sharpness_loss"]
    if "chamfer_loss" in losses:
        losses["loss"] = losses["loss"] + losses["chamfer_loss"]
    if "peak_loss" in losses:
        losses["loss"] = losses["loss"] + losses["peak_loss"]
    if "normal_loss" in losses:
        losses["loss"] = losses["loss"] + losses["normal_loss"]

    # ---- Model auxiliary losses (physics-informed branches) ----
    if model is not None and hasattr(model, 'aux_losses') and model.aux_losses:
        for aux_name, (aux_val, aux_weight) in model.aux_losses.items():
            if torch.is_tensor(aux_val) and aux_weight > 0:
                losses["loss"] = losses["loss"] + aux_val * aux_weight
                losses[f"aux_{aux_name}"] = aux_val.item()

    # ---- Monitoring metrics (no grad) ----
    with torch.no_grad():
        rmse = _rmse(pred_real, label_real)
        inactive_mask = (~active_mask) if active_mask is not None else None

        for i, fname in enumerate(fields):
            pred_ch_real = pred_real[..., i:i + 1]
            gt_ch_real = label_real[..., i:i + 1]

            rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)

            losses[f"L2_{fname}"] = rel_l2_val
            losses['mean_l2'] += rel_l2_val / num_channels
            losses[f"RMSE_{fname}"] = rmse[i].item()

            if active_mask is not None:
                ch_active = active_mask[..., i:i + 1]
                ch_inactive = inactive_mask[..., i:i + 1]
                losses[f"active_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_active)
                losses[f"inactive_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_inactive)
                losses[f"active_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_active)
                losses[f"inactive_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_inactive)

        if active_mask is not None:
            valid_active = [losses[f"active_L2_{f}"] for f in fields
                            if not math.isnan(losses.get(f"active_L2_{f}", float('nan')))]
            valid_inactive = [losses[f"inactive_L2_{f}"] for f in fields
                              if not math.isnan(losses.get(f"inactive_L2_{f}", float('nan')))]
            losses["active_mean_l2"] = sum(valid_active) / len(valid_active) if valid_active else float('nan')
            losses["inactive_mean_l2"] = sum(valid_inactive) / len(valid_inactive) if valid_inactive else float('nan')
            losses["active_loss"] = _masked_mse(pred_fp32, label_fp32, active_mask)
            losses["inactive_loss"] = _masked_mse(pred_fp32, label_fp32, inactive_mask)

    losses["each_l2"] = _each_l2(pred_real, label_real)

    return losses


# ---------------------------------------------------------------------------
# Validation loss (metrics only, no gradient)
# ---------------------------------------------------------------------------

def get_val_loss(fields, predict_hat, state, normalizer, active_mask=None):
    """Compute validation metrics (no gradient loss, no backprop)."""
    num_channels = float(len(fields))

    pred_fp32 = predict_hat.detach().float()
    label_fp32 = state.detach().float()

    pred_real = normalizer.denormalize(pred_fp32)
    state_real = normalizer.denormalize(label_fp32)

    losses = {'mean_l2': 0}

    rmse = _rmse(pred_real, state_real)
    inactive_mask = (~active_mask) if active_mask is not None else None

    for i, fname in enumerate(fields):
        pred_ch_real = pred_real[..., i:i + 1]
        gt_ch_real = state_real[..., i:i + 1]

        rel_l2_val = _relative_l2(pred_ch_real, gt_ch_real)

        losses[f"L2_{fname}"] = rel_l2_val
        losses['mean_l2'] += rel_l2_val / num_channels
        losses[f"RMSE_{fname}"] = rmse[i].item()

        if active_mask is not None:
            ch_active = active_mask[..., i:i + 1]
            ch_inactive = inactive_mask[..., i:i + 1]
            losses[f"active_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_active)
            losses[f"inactive_L2_{fname}"] = _masked_relative_l2(pred_ch_real, gt_ch_real, ch_inactive)
            losses[f"active_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_active)
            losses[f"inactive_RMSE_{fname}"] = _masked_rmse(pred_ch_real, gt_ch_real, ch_inactive)

    if active_mask is not None:
        valid_active = [losses[f"active_L2_{f}"] for f in fields
                        if not math.isnan(losses.get(f"active_L2_{f}", float('nan')))]
        valid_inactive = [losses[f"inactive_L2_{f}"] for f in fields
                          if not math.isnan(losses.get(f"inactive_L2_{f}", float('nan')))]
        losses["active_mean_l2"] = sum(valid_active) / len(valid_active) if valid_active else float('nan')
        losses["inactive_mean_l2"] = sum(valid_inactive) / len(valid_inactive) if valid_inactive else float('nan')

    losses["each_l2"] = _each_l2(pred_real, state_real)

    return losses
