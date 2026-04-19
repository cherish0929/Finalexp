"""
src/model — neural-operator model architectures for PhysGTO experiments.

Public API (model_name → class):
    "PhysGTO"                  → physgto.Model
    "gto_attnres_multi"        → physgto_attnres_multi.Model
    "gto_attnres_multi_v3"     → physgto_attnres_multi_v3.Model

Contrast models (model_name → class):
    "MGN"                      → contrast/mgn_model.Model
    "GNOT"                     → contrast/gnot_model.Model
    "Transolver"               → contrast/transolver_model.Model
    "GraphViT"                 → contrast/baseline_models.GraphViTModel
    "FNO3D"                    → contrast/baseline_models.FNO3DModel   (需要 grid_shape)
    "UNet3D"                   → contrast/baseline_models.UNet3DModel  (需要 grid_shape)
    "MeltPoolResNet"           → contrast/lpbf_baseline_models.MeltPoolResNet (需要 grid_shape)
    "ConvLSTMModel"            → contrast/lpbf_baseline_models.ConvLSTMModel  (需要 grid_shape)
    "ResNet3DModel"            → contrast/lpbf_baseline_models.ResNet3DModel  (需要 grid_shape)

Convenience helpers:
    build_model(model_cfg, cond_dim, default_dt, device, grid_shape=None) → nn.Module
    GRID_MODELS  — set of model names that require grid_shape
"""

from src.model.physgto import Model as PhysGTO
from src.model.physgto_attnres_multi import Model as GTO_attnres_multi
from src.model.physgto_attnres_multi_v3 import Model as GTO_attnres_multi_v3

# --------------------------------------------------------------------------- #
# PhysGTO 家族注册表
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "PhysGTO":               PhysGTO,
    "gto_attnres_multi":     GTO_attnres_multi,
    "gto_attnres_multi_v3":  GTO_attnres_multi_v3,
}

# 需要 grid_shape 的模型（基于均匀体素网格的卷积/FNO 模型）
GRID_MODELS = {"FNO3D", "UNet3D", "MeltPoolResNet", "ConvLSTMModel", "ResNet3DModel"}

__all__ = list(MODEL_REGISTRY.keys()) + ["MODEL_REGISTRY", "GRID_MODELS", "build_model"]


def build_model(model_cfg: dict, cond_dim: int, default_dt: float, device,
                grid_shape=None):
    """
    Instantiate a model from a config dict and move it to *device*.

    Args:
        model_cfg:   dict — the ``model`` section of a JSON config
        cond_dim:    condition feature dimension (inferred from dataset)
        default_dt:  default time-step (inferred from dataset)
        device:      torch.device or str
        grid_shape:  (D, H, W) tuple — required for grid-based contrast models
                     (FNO3D, UNet3D, MeltPoolResNet, ConvLSTMModel, ResNet3DModel)

    Returns:
        nn.Module on *device*, in eval mode (call .train() before training)

    Raises:
        ValueError if model_cfg["name"] is not recognised.
    """
    import torch.nn as nn

    model_name = model_cfg.get("name", "PhysGTO")

    # ── PhysGTO 家族 ──────────────────────────────────────────────────────────
    if model_name in MODEL_REGISTRY:
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

        # AttnRes 家族额外参数
        if model_name in ("gto_attnres_multi", "gto_attnres_multi_v2",
                          "gto_res_attnres", "gto_attnres_multi_v3", "gto_attnres_max"):
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
            kwargs["n_latent_cross"]   = model_cfg.get("n_latent_cross", 2)
            kwargs["gnn_light_ratio"]  = model_cfg.get("gnn_light_ratio", 0.5)
            kwargs["layer_scale_init"] = model_cfg.get("layer_scale_init", 1e-2)

        return ModelClass(**kwargs).to(device)

    # ── 对比模型 ──────────────────────────────────────────────────────────────
    return _build_contrast_model(model_cfg, cond_dim, default_dt, device, grid_shape)


def _build_contrast_model(model_cfg: dict, cond_dim: int, default_dt: float,
                           device, grid_shape=None):
    """
    实例化对比实验模型并移至 device。

    支持的 model_name:
        MGN, GNOT, Transolver, GraphViT,
        FNO3D, UNet3D, MeltPoolResNet, ConvLSTMModel, ResNet3DModel

    grid_shape: (D, H, W) — 网格模型必须提供。
    """
    from src.contrast.mgn_model import Model as MGN
    from src.contrast.gnot_model import Model as GNOT
    from src.contrast.transolver_model import Model as Transolver
    from src.contrast.baseline_models import GraphViTModel, FNO3DModel, UNet3DModel
    from src.contrast.lpbf_baseline_models import MeltPoolResNet, ConvLSTMModel, ResNet3DModel

    model_name  = model_cfg.get("name")
    space_size  = model_cfg.get("space_size", 3)
    pos_enc_dim = model_cfg.get("pos_enc_dim", 5)
    in_dim      = model_cfg.get("in_dim", 4)
    out_dim     = model_cfg.get("out_dim", 4)
    enc_dim     = model_cfg.get("enc_dim", 128)
    N_block     = model_cfg.get("N_block", 4)
    n_head      = model_cfg.get("n_head", 4)
    n_token     = model_cfg.get("n_token", 64)
    dt          = model_cfg.get("dt", default_dt)

    # ── 图/注意力模型（无网格假设）────────────────────────────────────────────
    if model_name == "MGN":
        model = MGN(
            space_size=space_size, pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
            N_block=N_block, in_dim=in_dim, out_dim=out_dim, enc_dim=enc_dim,
            n_head=n_head, n_token=n_token, dt=dt,
        )

    elif model_name == "GNOT":
        # n_token 在 GNOT 中表示 MoE 专家数
        model = GNOT(
            space_size=space_size, pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
            N_block=N_block, in_dim=in_dim, out_dim=out_dim, enc_dim=enc_dim,
            n_head=n_head, n_token=n_token, dt=dt,
        )

    elif model_name == "Transolver":
        # n_token 在 Transolver 中表示物理切片数 M
        model = Transolver(
            space_size=space_size, pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
            N_block=N_block, in_dim=in_dim, out_dim=out_dim, enc_dim=enc_dim,
            n_head=n_head, n_token=n_token, dt=dt,
        )

    elif model_name == "GraphViT":
        # N_block 拆分为 GNN 层数 + Transformer 层数
        n_gnn_layers  = model_cfg.get("n_gnn_layers",  max(1, N_block // 2))
        n_attn_layers = model_cfg.get("n_attn_layers", max(1, N_block // 2))
        model = GraphViTModel(
            space_size=space_size, in_dim=in_dim, out_dim=out_dim, enc_dim=enc_dim,
            pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
            n_gnn_layers=n_gnn_layers, n_attn_layers=n_attn_layers,
            n_heads=n_head, dt=dt,
        )

    # ── 网格模型（需要 grid_shape）────────────────────────────────────────────
    elif model_name in GRID_MODELS:
        if grid_shape is None:
            raise ValueError(
                f"Model '{model_name}' requires grid_shape=(D,H,W) but none was provided. "
                "Pass grid_shape to build_model() or get_model()."
            )

        if model_name == "FNO3D":
            modes   = tuple(model_cfg.get("modes", [8, 8, 8]))
            n_layers = model_cfg.get("n_layers", N_block)
            model = FNO3DModel(
                grid_shape=grid_shape, space_size=space_size,
                in_dim=in_dim, out_dim=out_dim, enc_dim=enc_dim,
                pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
                modes=modes, n_layers=n_layers, dt=dt,
            )

        elif model_name == "UNet3D":
            base_ch  = model_cfg.get("base_ch", 32)
            n_levels = model_cfg.get("n_levels", 3)
            model = UNet3DModel(
                grid_shape=grid_shape, space_size=space_size,
                in_dim=in_dim, out_dim=out_dim, base_ch=base_ch,
                pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
                n_levels=n_levels, dt=dt,
            )

        elif model_name == "MeltPoolResNet":
            base_ch = model_cfg.get("base_ch", 32)
            n_res   = model_cfg.get("n_res", 2)
            model = MeltPoolResNet(
                grid_shape=grid_shape, space_size=space_size,
                in_dim=in_dim, out_dim=out_dim, base_ch=base_ch,
                pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
                n_res=n_res, dt=dt,
            )

        elif model_name == "ConvLSTMModel":
            base_ch     = model_cfg.get("base_ch", 32)
            n_gru_steps = model_cfg.get("n_gru_steps", 2)
            model = ConvLSTMModel(
                grid_shape=grid_shape, space_size=space_size,
                in_dim=in_dim, out_dim=out_dim, base_ch=base_ch,
                pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
                n_gru_steps=n_gru_steps, dt=dt,
            )

        elif model_name == "ResNet3DModel":
            ch        = model_cfg.get("ch", 64)
            n_blocks  = model_cfg.get("n_blocks", 8)
            dilations = tuple(model_cfg.get("dilations", [1, 2, 4]))
            model = ResNet3DModel(
                grid_shape=grid_shape, space_size=space_size,
                in_dim=in_dim, out_dim=out_dim, ch=ch,
                pos_enc_dim=pos_enc_dim, cond_dim=cond_dim,
                n_blocks=n_blocks, dilations=dilations, dt=dt,
            )

    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"PhysGTO family: {list(MODEL_REGISTRY.keys())}. "
            f"Contrast models: MGN, GNOT, Transolver, GraphViT, "
            f"FNO3D, UNet3D, MeltPoolResNet, ConvLSTMModel, ResNet3DModel."
        )

    return model.to(device)
