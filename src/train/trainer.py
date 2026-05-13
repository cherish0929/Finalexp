"""train_v2 — training loop with configurable grad_loss_weight, NaN/spike guards, EMA."""

import torch
from tqdm import tqdm
from torch.amp import autocast

from .autoregressive import _autoregressive_lpbf
from .losses import get_train_loss
from .metrics import _REGION_PREFIXES, _REGION_MEANS, _init_region_agg, _accumulate_region, _finalize_region


def train_v2(args, model, train_dataloader, optim, device, normalizer, ema=None, ckpt_threshold=None):
    """train() with configurable grad_loss_weight instead of hardcoded 8.0."""
    horizon = args.data.get("horizon_train", 1) if isinstance(args.data, dict) else getattr(args, "horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp = args.train.get("use_amp", False)
    check_point = ckpt_threshold if ckpt_threshold is not None else args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})
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

    _use_lpbf = model_name == "gto_lpbf"
    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max", "gto_lpbf")
    _needs_node_pos_loss = weight_loss.get("chamfer", False) or weight_loss.get("normal", False) or weight_loss.get("peak", False)
    _has_aux = model_name == "gto_lpbf"

    pbar = tqdm(train_dataloader, desc="  Train", unit="bt", leave=True, ncols=120, colour='green')
    _n_skip = 0
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
                if _use_lpbf:
                    predict_hat = _autoregressive_lpbf(
                        model, state[:, 0], node_pos, edges, time_seq,
                        spatial_inform, conditions, dt, check_point, batch, device)
                elif _use_spatial:
                    predict_hat = model.autoregressive(
                        state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
                else:
                    predict_hat = model.autoregressive(
                        state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

                costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss,
                                      active_mask=active_mask,
                                      node_pos=node_pos if _needs_node_pos_loss else None,
                                      model=model if _has_aux else None)

            loss = costs["loss"]

            # NaN guard / Loss spike guard
            _skip = False
            if not torch.isfinite(loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                _n_skip += 1
                pbar.set_postfix({"Loss": "NaN/skip", "skip": _n_skip})
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                _n_skip += 1
                pbar.set_postfix({"Loss": "grad NaN", "skip": _n_skip})
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)
        else:
            if _use_lpbf:
                predict_hat = _autoregressive_lpbf(
                    model, state[:, 0], node_pos, edges, time_seq,
                    spatial_inform, conditions, dt, check_point, batch, device)
            elif _use_spatial:
                predict_hat = model.autoregressive(
                    state[:, 0], node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point)
            else:
                predict_hat = model.autoregressive(
                    state[:, 0], node_pos, edges, time_seq, conditions, dt, check_point)

            costs = get_train_loss(fields, predict_hat, state[:, 1:], normalizer, weight_loss,
                                  active_mask=active_mask,
                                  node_pos=node_pos if _needs_node_pos_loss else None,
                                  model=model if _has_aux else None)
            loss = costs["loss"]

            _skip = False
            if not torch.isfinite(loss):
                _skip = True
            else:
                _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
                if _cur_avg is not None and loss.item() > 10 * _cur_avg:
                    _skip = True
            if _skip:
                _n_skip += 1
                pbar.set_postfix({"Loss": "NaN/skip", "skip": _n_skip})
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.train.get("grad_clip", 1.0))
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                _n_skip += 1
                pbar.set_postfix({"Loss": "grad NaN", "skip": _n_skip})
                optim.zero_grad()
                del predict_hat, costs, loss
                torch.cuda.empty_cache()
                continue
            optim.step()
            optim.zero_grad()
            if ema is not None:
                ema.update(model)

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
        _postfix = {"Loss": f"{avg_loss:.4e}"}
        if _n_skip > 0:
            _postfix["skip"] = _n_skip
        pbar.set_postfix(_postfix)

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
