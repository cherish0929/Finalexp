"""Metric functions: L2, RMSE, masked variants, and region aggregation helpers."""

import math
import torch


# ---------------------------------------------------------------------------
# Relative L2 error (denormalized data)
# ---------------------------------------------------------------------------

def _relative_l2(pred, target):
    """Relative L2 error, returns [batch] tensor."""
    error = pred - target
    norm_error = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-6)
    norm_error_channel = torch.mean(norm_error, dim=-1)
    norm_error_time = torch.mean(norm_error_channel, dim=-1)
    return norm_error_time


def _each_l2(pred, target):
    """Per-timestep relative L2."""
    t_step = target.shape[1]
    losses_each_t = torch.zeros(t_step, device=pred.device)
    for t in range(t_step):
        error = pred[:, t] - target[:, t]
        norm_error = torch.norm(error, dim=-2) / (torch.norm(target[:, t], dim=-2) + 1e-6)
        norm_error_channel = torch.mean(norm_error, dim=-1)
        norm_error_batch = torch.mean(norm_error_channel, dim=0)
        losses_each_t[t] = norm_error_batch
    return losses_each_t


def _rmse(pred, target):
    """RMSE over batch, time, and spatial dimensions."""
    diff = pred - target
    mse = torch.mean(diff ** 2, dim=[0, 1, 2])
    return torch.sqrt(mse)


# ---------------------------------------------------------------------------
# Region-masked metrics
# ---------------------------------------------------------------------------

def _masked_relative_l2(pred, target, mask):
    """
    Relative L2 error over masked spatial points for a single channel.
    Args:
        pred, target: [B, T, N, 1] denormalized
        mask: [B, T, N, 1] bool
    Returns:
        float scalar, or float('nan') if mask has zero True entries.
    """
    if mask.sum() == 0:
        return float('nan')
    error = (pred - target) * mask.float()
    target_masked = target * mask.float()
    error_norm = torch.norm(error, dim=-2)
    target_norm = torch.norm(target_masked, dim=-2)
    rel = error_norm / (target_norm + 1e-6)
    return rel.mean().item()


def _masked_rmse(pred, target, mask):
    """
    RMSE over masked spatial points for a single channel.
    Args:
        pred, target: [B, T, N, 1] denormalized
        mask: [B, T, N, 1] bool
    Returns:
        float scalar, or float('nan') if mask is empty.
    """
    count = mask.sum()
    if count == 0:
        return float('nan')
    diff = (pred - target) * mask.float()
    mse = (diff ** 2).sum() / count
    return torch.sqrt(mse).item()


def _masked_mse(pred, target, mask):
    """
    MSE over masked region (monitoring only, not for backprop).
    Args:
        pred, target: [B, T, N, C] (normalized)
        mask: [B, T, N, C] bool
    Returns:
        float scalar.
    """
    count = mask.sum()
    if count == 0:
        return 0.0
    diff = (pred - target) * mask.float()
    return ((diff ** 2).sum() / count).item()


# ---------------------------------------------------------------------------
# Region metric aggregation
# ---------------------------------------------------------------------------

_REGION_PREFIXES = ("active_L2", "inactive_L2", "active_RMSE", "inactive_RMSE")
_REGION_MEANS = ("active_mean_l2", "inactive_mean_l2")


def _init_region_agg(agg, fields):
    """Initialize region metric accumulators in the agg dict."""
    for fname in fields:
        for prefix in _REGION_PREFIXES:
            agg[f"{prefix}_{fname}"] = 0.0
            agg[f"{prefix}_{fname}_cnt"] = 0
    for key in _REGION_MEANS:
        agg[key] = 0.0
        agg[f"{key}_cnt"] = 0
    agg["active_loss"] = 0.0
    agg["inactive_loss"] = 0.0


def _accumulate_region(agg, costs, batch_num, fields, include_loss=True):
    """Accumulate region metrics from one batch, skipping NaN values."""
    for fname in fields:
        for prefix in _REGION_PREFIXES:
            val = costs.get(f"{prefix}_{fname}")
            if val is not None and not math.isnan(val):
                agg[f"{prefix}_{fname}"] += val * batch_num
                agg[f"{prefix}_{fname}_cnt"] += batch_num
    for key in _REGION_MEANS:
        val = costs.get(key)
        if val is not None and not math.isnan(val):
            agg[key] += val * batch_num
            agg[f"{key}_cnt"] += batch_num
    if include_loss:
        agg["active_loss"] += costs.get("active_loss", 0.0) * batch_num
        agg["inactive_loss"] += costs.get("inactive_loss", 0.0) * batch_num


def _finalize_region(agg, fields, total_num, include_loss=True):
    """Average region accumulators and remove count keys."""
    for fname in fields:
        for prefix in _REGION_PREFIXES:
            cnt = agg.pop(f"{prefix}_{fname}_cnt", 0)
            agg[f"{prefix}_{fname}"] = agg[f"{prefix}_{fname}"] / cnt if cnt > 0 else float('nan')
    for key in _REGION_MEANS:
        cnt = agg.pop(f"{key}_cnt", 0)
        agg[key] = agg[key] / cnt if cnt > 0 else float('nan')
    if include_loss:
        agg["active_loss"] = agg["active_loss"] / total_num if total_num > 0 else 0.0
        agg["inactive_loss"] = agg["inactive_loss"] / total_num if total_num > 0 else 0.0
