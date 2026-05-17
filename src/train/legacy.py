"""Legacy train and validate loops (used by main.py, evaluate.py)."""

import torch
import random
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from .autoregressive import _autoregressive_lpbf
from .losses import get_train_loss, get_val_loss
from .metrics import _REGION_PREFIXES, _REGION_MEANS, _init_region_agg, _accumulate_region, _finalize_region


def train(args, model, train_dataloader, optim, device, normalizer):
    horizon = args.data.get("horizon_train", 1) if isinstance(args.data, dict) else getattr(args, "horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp, check_point = args.train.get("use_amp", False), args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})

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
    if use_amp:
        scaler = GradScaler('cuda')

    pbar = tqdm(train_dataloader, desc="  Train", unit="bt", leave=True, ncols=120, colour='green')
    for batch in pbar:
        dt = batch['dt'].to(device)
        state = batch["state"].to(device)
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq = batch["time_seq"].to(device)
        conditions = batch["conditions"].to(device).float()
        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch['grid_shape'].numpy()

        active_mask = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask[:, 1:].to(device)
            if not has_region:
                _init_region_agg(agg, fields)
                has_region = True

        batch_num = state.shape[0]

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)

            costs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()
        else:
            predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
            costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss, active_mask=active_mask)
            costs["loss"].backward()
            optim.step()
            optim.zero_grad()

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
        pbar.set_postfix({"Loss": f"{avg_loss :.4e}"})

    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in ("active_loss", "inactive_loss") and key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=True)
    return agg


def validate(args, model, val_dataloader, device, normalizer, epoch):
    horizon = args.data.get("horizon_test", 1) if isinstance(args.data, dict) else getattr(args, "horizon_test", 1)
    fields = args.data.get("fields", ["T"])
    use_amp, check_point = args.train.get("use_amp", False), args.train.get("check_point", False)
    model_name = args.model.get("name", "PhysGTO")
    _use_lpbf = model_name in ("gto_lpbf", "gto_lpbf_v2")
    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max", "gto_lpbf", "gto_lpbf_v2")

    agg = {}
    for key in ["L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(horizon, device=device)
    agg["num"] = 0
    has_region = False

    model.eval()
    normalizer.to(device)

    num_batches = len(val_dataloader)
    num_viz = min(2, num_batches)
    viz_batch_indices = set(random.sample(range(num_batches), num_viz))

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="  Valid", unit="bt", leave=False, ncols=120, colour='yellow')
        for i, batch in enumerate(pbar):
            dt = batch['dt'].to(device)
            state = batch["state"].to(device)
            node_pos = batch["node_pos"].to(device)
            edges = batch["edges"].to(device)
            time_seq = batch["time_seq"].to(device)
            if _use_spatial:
                spatial_inform = batch["spatial_inform"].to(device)
            conditions = batch["conditions"].to(device).float()

            active_mask = batch.get("active_mask")
            if active_mask is not None:
                active_mask = active_mask[:, 1:].to(device)
                if not has_region:
                    _init_region_agg(agg, fields)
                    has_region = True

            batch_num = state.shape[0]

            if use_amp:
                with autocast("cuda", dtype=torch.bfloat16):
                    if _use_lpbf:
                        predict_hat = _autoregressive_lpbf(model, state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point, batch, device)
                    elif _use_spatial:
                        predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
                    else:
                        predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                    costs = get_val_loss(fields, predict_hat, state[:, 1:], normalizer, active_mask=active_mask)
            else:
                if _use_lpbf:
                    predict_hat = _autoregressive_lpbf(model, state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point, batch, device)
                elif _use_spatial:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
                else:
                    predict_hat = model.autoregressive(state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)
                costs = get_val_loss(fields, predict_hat, state[:, 1:], normalizer, active_mask=active_mask)

            for fname in fields:
                agg[f"L2_{fname}"] += costs[f"L2_{fname}"].mean().item() * batch_num
                agg[f"RMSE_{fname}"] += costs[f"RMSE_{fname}"] * batch_num
            agg["mean_l2"] += costs["mean_l2"].mean().item() * batch_num
            agg["each_l2"] += costs["each_l2"] * batch_num
            agg["num"] += batch_num

            if has_region:
                _accumulate_region(agg, costs, batch_num, fields, include_loss=False)

    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=False)
    return agg
