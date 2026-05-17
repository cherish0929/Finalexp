"""Pushforward training: extend rollout beyond horizon_train by extra steps."""

import torch
from tqdm import tqdm
from torch.amp import autocast

from .autoregressive import _autoregressive_lpbf
from .losses import get_train_loss
from .metrics import _REGION_PREFIXES, _REGION_MEANS, _init_region_agg, _accumulate_region, _finalize_region


def _pf_forward_and_loss(model, state, node_pos, edges, time_seq, conditions,
                         spatial_inform, dt, check_point, batch, device,
                         use_amp, _use_lpbf, _use_spatial, fields, normalizer,
                         weight_loss, base_horizon, T_pf, active_mask,
                         _needs_node_pos_loss, _has_aux):
    """Run pushforward forward pass + base/extra loss. Returns (predict_hat, costs, total_loss)."""
    base_mask = active_mask[:, :base_horizon] if active_mask is not None else None

    def _run():
        if _use_lpbf:
            return _autoregressive_lpbf(
                model, state[:, 0], node_pos, edges, time_seq[:, :T_pf],
                spatial_inform, conditions, dt, check_point, batch, device)
        elif _use_spatial:
            return model.autoregressive(
                state[:, 0], node_pos, edges, time_seq[:, :T_pf], spatial_inform, conditions, dt, check_point)
        else:
            return model.autoregressive(
                state[:, 0], node_pos, edges, time_seq[:, :T_pf], conditions, dt, check_point)

    def _losses(predict_hat):
        costs = get_train_loss(fields, predict_hat[:, :base_horizon], state[:, 1:base_horizon + 1],
                               normalizer, weight_loss,
                               active_mask=base_mask,
                               node_pos=node_pos if _needs_node_pos_loss else None,
                               model=model if _has_aux else None)
        loss_base = costs["loss"]
        if T_pf > base_horizon and state.shape[1] > base_horizon + 1:
            T_extra = min(T_pf, state.shape[1] - 1)
            pf_mask = active_mask[:, base_horizon:T_extra] if active_mask is not None else None
            costs_pf = get_train_loss(
                fields, predict_hat[:, base_horizon:T_extra],
                state[:, base_horizon + 1:T_extra + 1], normalizer, weight_loss,
                active_mask=pf_mask,
                node_pos=node_pos if _needs_node_pos_loss else None,
                model=model if _has_aux else None)
            total_loss = loss_base + 0.5 * costs_pf["loss"]
        else:
            total_loss = loss_base
        return costs, total_loss

    if use_amp:
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            predict_hat = _run()
            costs, total_loss = _losses(predict_hat)
    else:
        predict_hat = _run()
        costs, total_loss = _losses(predict_hat)

    return predict_hat, costs, total_loss


def train_pushforward(args, model, train_dataloader, optim, device, normalizer, extra_steps, ema=None, ckpt_threshold=None):
    """
    Pushforward training: extend rollout beyond horizon_train by extra_steps.
    Only backpropagate through the extra steps (the model is already good at
    the original horizon). This forces the model to learn to correct its own errors.
    """
    base_horizon = args.data.get("horizon_train", 1)
    fields = args.data.get("fields", ["T"])
    use_amp = args.train.get("use_amp", False)
    check_point = ckpt_threshold if ckpt_threshold is not None else args.train.get("check_point", False)
    weight_loss = args.train.get("weight_loss", {"enable": False})
    model_name = args.model.get("name", "PhysGTO")
    grad_clip = args.train.get("grad_clip", 1.0)

    agg = {}
    for key in ["loss", "L2", "mean_l2", "RMSE"]:
        if key == "L2" or key == "RMSE":
            for fname in fields:
                agg[f"{key}_{fname}"] = 0.0
        else:
            agg[key] = 0.0
    agg["each_l2"] = torch.zeros(base_horizon, device=device)
    agg["num"] = 0
    agg["value_loss"] = 0.0
    agg["grad_loss"] = 0.0
    has_region = False

    model.train()
    normalizer.to(device)

    _use_lpbf = model_name in ("gto_lpbf", "gto_lpbf_v2")
    _use_spatial = model_name in ("PhysGTO_v2", "gto_attnres_multi_v3", "gto_attnres_max", "gto_lpbf", "gto_lpbf_v2")
    _needs_node_pos_loss = weight_loss.get("chamfer", False) or weight_loss.get("normal", False) or weight_loss.get("peak", False)
    _has_aux = model_name in ("gto_lpbf", "gto_lpbf_v2")

    _fwd_kwargs = dict(
        use_amp=use_amp, _use_lpbf=_use_lpbf, _use_spatial=_use_spatial,
        fields=fields, normalizer=normalizer, weight_loss=weight_loss,
        _needs_node_pos_loss=_needs_node_pos_loss, _has_aux=_has_aux,
    )

    pbar = tqdm(train_dataloader, desc="  Train(PF)", unit="bt", leave=True, ncols=120, colour='cyan')
    _n_skip = 0
    _n_oom_retry = 0

    for batch in pbar:
        dt = batch['dt'].to(device)
        state_cpu = batch["state"]
        node_pos = batch["node_pos"].to(device)
        edges = batch["edges"].to(device)
        time_seq_cpu = batch["time_seq"]
        spatial_inform = batch["spatial_inform"].to(device) if _use_spatial else None
        conditions = batch["conditions"].to(device).float()
        if weight_loss.get("gradient", False):
            weight_loss["grid_shape"] = batch['grid_shape'].numpy()

        active_mask = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask[:, 1:].to(device)
            if not has_region:
                _init_region_agg(agg, fields)
                has_region = True

        batch_num = state_cpu.shape[0]
        T_total = time_seq_cpu.shape[1]
        T_pf = min(base_horizon + extra_steps, T_total)

        state = state_cpu[:, :T_pf + 1].to(device)
        time_seq = time_seq_cpu[:, :T_pf].to(device)

        # Forward + loss, with OOM retry using full gradient checkpointing
        predict_hat = costs = total_loss = None
        try:
            predict_hat, costs, total_loss = _pf_forward_and_loss(
                model, state, node_pos, edges, time_seq, conditions,
                spatial_inform, dt, check_point, batch, device,
                base_horizon=base_horizon, T_pf=T_pf, active_mask=active_mask,
                **_fwd_kwargs)
        except torch.cuda.OutOfMemoryError:
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if check_point is not True:
                _n_oom_retry += 1
                try:
                    predict_hat, costs, total_loss = _pf_forward_and_loss(
                        model, state, node_pos, edges, time_seq, conditions,
                        spatial_inform, dt, True, batch, device,
                        base_horizon=base_horizon, T_pf=T_pf, active_mask=active_mask,
                        **_fwd_kwargs)
                except torch.cuda.OutOfMemoryError:
                    optim.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    _n_skip += 1
                    pbar.set_postfix({"Loss": "OOM", "skip": _n_skip})
                    continue
            else:
                _n_skip += 1
                pbar.set_postfix({"Loss": "OOM", "skip": _n_skip})
                continue

        # NaN / spike guard
        _skip = False
        if not torch.isfinite(total_loss):
            _skip = True
        else:
            _cur_avg = agg["loss"] / agg["num"] if agg["num"] > 0 else None
            if _cur_avg is not None and total_loss.item() > 10 * _cur_avg:
                _skip = True
        if _skip:
            _n_skip += 1
            optim.zero_grad()
            del predict_hat, costs, total_loss
            torch.cuda.empty_cache()
            pbar.set_postfix({"Loss": "NaN/skip", "skip": _n_skip})
            continue

        # Backward with OOM retry
        try:
            total_loss.backward()
        except torch.cuda.OutOfMemoryError:
            optim.zero_grad(set_to_none=True)
            del predict_hat, costs, total_loss
            torch.cuda.empty_cache()
            if check_point is not True:
                _n_oom_retry += 1
                try:
                    predict_hat, costs, total_loss = _pf_forward_and_loss(
                        model, state, node_pos, edges, time_seq, conditions,
                        spatial_inform, dt, True, batch, device,
                        base_horizon=base_horizon, T_pf=T_pf, active_mask=active_mask,
                        **_fwd_kwargs)
                    total_loss.backward()
                except torch.cuda.OutOfMemoryError:
                    optim.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    _n_skip += 1
                    pbar.set_postfix({"Loss": "OOM", "skip": _n_skip})
                    continue
            else:
                _n_skip += 1
                pbar.set_postfix({"Loss": "OOM", "skip": _n_skip})
                continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            _n_skip += 1
            optim.zero_grad()
            del predict_hat, costs, total_loss
            torch.cuda.empty_cache()
            pbar.set_postfix({"Loss": "grad NaN", "skip": _n_skip})
            continue
        optim.step()
        optim.zero_grad()
        if ema is not None:
            ema.update(model)

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
        if _n_oom_retry > 0:
            _postfix["oom_retry"] = _n_oom_retry
        pbar.set_postfix(_postfix)

    if agg["num"] == 0:
        raise RuntimeError(
            "train_pushforward: all batches were skipped. "
            "Loss is NaN or Inf — check for gradient explosion or bad input data."
        )
    for key, value in agg.items():
        if key != "each_l2" and key != "num" and not key.endswith("_cnt"):
            if key not in ("active_loss", "inactive_loss") and key not in _REGION_MEANS and not any(key.startswith(p + "_") for p in _REGION_PREFIXES):
                agg[key] = value / agg["num"]

    agg["each_l2"] = (agg["each_l2"] / agg["num"]).cpu()
    if has_region:
        _finalize_region(agg, fields, agg["num"], include_loss=True)
    if _n_oom_retry > 0:
        print(f"  [OOM] {_n_oom_retry} batches retried with full gradient checkpointing")
    return agg
