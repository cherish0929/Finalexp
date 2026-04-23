"""
src/model — neural-operator model architectures for PhysGTO experiments.

Public API (model_name → class):
    "PhysGTO"                  → physgto.Model
    "PhysGTO_v2"               → physgto_v2.Model
    "gto_res"                  → physgto_res.Model
    "gto_lnn"                  → gto_lnn.Model
    "gto_attnres_multi"        → physgto_attnres_multi.Model
    "gto_attnres_multi_v2"     → physgto_attnres_multi_v2.Model
    "gto_res_attnres"          → physgto_res_attnres.Model
    "gto_attnres_multi_v3"     → physgto_attnres_multi_v3.Model
    "gto_attnres_max"          → physgto_attnres_max.Model  ← NEW

Convenience helper
    build_model(model_cfg, cond_dim, default_dt, device) → nn.Module
"""

from src.model.physgto import Model as PhysGTO
from src.model.physgto_v2 import Model as PhysGTO_v2
from src.model.physgto_res import Model as PhysGTO_res
from src.model.gto_lnn import Model as GTO_lnn
from src.model.physgto_attnres_multi import Model as GTO_attnres_multi
from src.model.physgto_attnres_multi_v2 import Model as GTO_attnres_multi_v2
from src.model.physgto_res_attnres import Model as GTO_res_attnres
from src.model.physgto_attnres_multi_v3 import Model as GTO_attnres_multi_v3
from src.model.physgto_attnres_max import Model as GTO_attnres_max
from src.model.physgto_lpbf import Model as GTO_lpbf

# --------------------------------------------------------------------------- #
# Registry: model_name (str used in JSON config) → Model class
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "PhysGTO":               PhysGTO,
    "PhysGTO_v2":            PhysGTO_v2,
    "gto_res":               PhysGTO_res,
    "gto_lnn":               GTO_lnn,
    "gto_attnres_multi":     GTO_attnres_multi,
    "gto_attnres_multi_v2":  GTO_attnres_multi_v2,
    "gto_res_attnres":       GTO_res_attnres,
    "gto_attnres_multi_v3":  GTO_attnres_multi_v3,
    "gto_attnres_max":       GTO_attnres_max,
    "gto_lpbf":              GTO_lpbf,
}

__all__ = list(MODEL_REGISTRY.keys()) + ["MODEL_REGISTRY", "build_model"]


def build_model(model_cfg: dict, cond_dim: int, default_dt: float, device):
    """
    Instantiate a model from a config dict and move it to *device*.

    Args:
        model_cfg:   dict — the ``model`` section of a JSON config
        cond_dim:    condition feature dimension (inferred from dataset)
        default_dt:  default time-step (inferred from dataset)
        device:      torch.device or str

    Returns:
        nn.Module on *device*, in eval mode (call .train() before training)

    Raises:
        ValueError if model_cfg["name"] is not in MODEL_REGISTRY.
    """
    import torch.nn as nn

    model_name = model_cfg.get("name", "PhysGTO")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[model_name]

    kwargs = dict(
        space_size=model_cfg.get("space_size", 3),
        pos_enc_dim=model_cfg.get("pos_enc_dim", 5),
        cond_dim=cond_dim,
        N_block=model_cfg.get("N_block", 4),
        in_dim=model_cfg.get("in_dim", 4),
        out_dim=model_cfg.get("out_dim", 4),
        enc_dim=model_cfg.get("enc_dim", 128),
        n_head=model_cfg.get("n_head", 4),
        n_token=model_cfg.get("n_token", 64),
        dt=model_cfg.get("dt", default_dt),
    )

    # AttnRes family
    if model_name in (
        "gto_attnres_multi", "gto_attnres_multi_v2",
        "gto_res_attnres", "gto_attnres_multi_v3", "gto_attnres_max",
    ):
        kwargs["n_fields"] = model_cfg.get("n_fields", model_cfg.get("in_dim", 2))
        kwargs["cross_attn_heads"] = model_cfg.get("cross_attn_heads", 4)

    if model_name in ("gto_attnres_multi_v2", "gto_res_attnres"):
        kwargs["attn_res_mode"] = model_cfg.get("attn_res_mode", "block_inter")

    if model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max"):
        kwargs["spatial_dim"] = model_cfg.get("spatial_dim", 10)
        kwargs["pos_x_boost"] = model_cfg.get("pos_x_boost", 2)

    if model_name in ("gto_attnres_multi_v3", "gto_attnres_max"):
        kwargs["n_latent"] = model_cfg.get("n_latent", 4)

    if model_name == "gto_attnres_max":
        kwargs["n_latent_cross"]     = model_cfg.get("n_latent_cross", 2)
        kwargs["gnn_light_ratio"]    = model_cfg.get("gnn_light_ratio", 0.5)
        kwargs["layer_scale_init"]   = model_cfg.get("layer_scale_init", 1e-2)
        kwargs["use_intra_attn_res"] = model_cfg.get("use_intra_attn_res", False)

    if model_name == "gto_lpbf":
        # fields is required — derive from data config if not in model config
        fields = model_cfg.get("fields", model_cfg.get("_fields", ["T"]))
        kwargs["fields"]          = fields
        kwargs["spatial_dim"]     = model_cfg.get("spatial_dim", 10)
        kwargs["pos_x_boost"]     = model_cfg.get("pos_x_boost", 2)
        kwargs["n_latent"]        = model_cfg.get("n_latent", 4)
        kwargs["n_fields"]        = model_cfg.get("n_fields", len(fields))
        kwargs["cross_attn_heads"]= model_cfg.get("cross_attn_heads", 4)
        kwargs["d_laser"]         = model_cfg.get("d_laser", 32)
        kwargs["ortho_weight"]    = model_cfg.get("ortho_weight", 0.01)
        kwargs["balance_weight"]  = model_cfg.get("balance_weight", 0.05)
        kwargs["sharp_weight"]    = model_cfg.get("sharp_weight", 0.01)
        kwargs["stepper_scheme"]  = model_cfg.get("stepper_scheme", "euler")

    return ModelClass(**kwargs).to(device)
