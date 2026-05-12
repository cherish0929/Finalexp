"""3D spatial gradient computation via first-order finite differences."""

import torch


def compute_spatial_gradient_3d(tensor_field, grid_shape):
    """
    Compute 3D spatial gradients (first-order finite differences).

    Args:
        tensor_field: [B, T, N, C] flattened tensor
        grid_shape: (Nx, Ny, Nz) 3D grid dimensions
    Returns:
        grad_x, grad_y, grad_z tensors
    """
    B, T, N, C = tensor_field.shape
    Nx, Ny, Nz = int(grid_shape[0][0]), int(grid_shape[0][1]), int(grid_shape[0][2])
    assert Nx * Ny * Nz == N, f"Grid shape {grid_shape} does not match node count {N}!"

    grid_field = tensor_field.view(B, T, Nz, Ny, Nx, C)

    grad_x = grid_field[:, :, :, :, 1:, :] - grid_field[:, :, :, :, :-1, :]
    grad_y = grid_field[:, :, :, 1:, :, :] - grid_field[:, :, :, :-1, :, :]
    grad_z = grid_field[:, :, 1:, :, :, :] - grid_field[:, :, :-1, :, :, :]

    return grad_x, grad_y, grad_z
