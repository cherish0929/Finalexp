"""
Backward-compatible shim — re-exports everything from the src.train package.

Existing imports like:
    from src.train import train, validate, get_train_loss
continue to work unchanged after the refactor.
"""

from src.train.autoregressive import _autoregressive_lpbf
from src.train.gradient import compute_spatial_gradient_3d
from src.train.helpers import _is_vof_field
from src.train.metrics import (
    _relative_l2,
    _each_l2,
    _rmse,
    _masked_relative_l2,
    _masked_rmse,
    _masked_mse,
    _REGION_PREFIXES,
    _REGION_MEANS,
    _init_region_agg,
    _accumulate_region,
    _finalize_region,
)
from src.train.losses import (
    _build_weight_mask,
    _compute_weighted_value_loss,
    _adapt_mask_to_gradient,
    _compute_gradient_loss,
    _compute_laplacian_loss,
    _compute_sharpness_loss,
    _compute_chamfer_loss,
    _compute_peak_loss,
    _compute_normal_consistency,
    get_train_loss,
    get_val_loss,
)
from src.train.legacy import train, validate
from src.train.trainer import train_v2
from src.train.pushforward import train_pushforward
