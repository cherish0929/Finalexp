"""LPBF-specific autoregressive helper: extract extras from batch and call model."""

import torch


def _autoregressive_lpbf(model, state0, node_pos, edges, time_seq, spatial_inform,
                          conditions, dt, check_point, batch, device):
    """Extract LPBF extras from batch and call model.autoregressive."""
    node_pos_abs = batch.get("node_pos_abs")
    laser_params = batch.get("laser_params")
    laser_traj = batch.get("laser_traj")
    abs_time_seq = batch.get("abs_time_seq")

    T = time_seq.shape[1]

    if laser_traj is not None:
        laser_traj = laser_traj[:, :T + 1].to(device)
    if abs_time_seq is not None:
        abs_time_seq = abs_time_seq[:, :T + 1].to(device)
    if node_pos_abs is not None:
        node_pos_abs = node_pos_abs.to(device)
    if laser_params is not None:
        laser_params = laser_params.to(device).float()

    return model.autoregressive(
        state0, node_pos, edges, time_seq, spatial_inform, conditions, dt, check_point,
        node_pos_abs=node_pos_abs, laser_params=laser_params,
        laser_traj=laser_traj, abs_time_seq=abs_time_seq,
    )
