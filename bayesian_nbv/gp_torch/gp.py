import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_nbv.gp_torch.kernels import AbstractKernel

@torch.compile
def solve_spd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b, L)

def compute_mean(
    x_query: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    kernel: AbstractKernel,
    var_noise: float
):
    """
    Compute the mean of the Gaussian Process at the query points.
    :math:`\mu(x_*) = k(x_*, X) K(X, X)^{-1} y`

    Args:
        x_query: Query points (shape: [n_query, input_dim]).
        x_data: Training data points (shape: [n_data, input_dim]).
        y_data: Training targets (shape: [n_data, output_dim]).
        kernel: Kernel function to compute covariance.
        var_noise: Noise level (variance).

    Returns:
        Mean at the query points (shape: [n_query, output_dim]).
    """
    k_xx = kernel(x_data, x_data) + var_noise * torch.eye(x_data.shape[0], device=x_data.device)  # [n_data, n_data]
    k_xq = kernel(x_data, x_query)  # [n_data, n_query]

    y_solved = solve_spd(k_xx, y_data)  # [n_data, output_dim]
    mean = k_xq.T @ y_solved  # [n_query, n_data] @ [n_data, output_dim] -> [n_query, output_dim]

    return mean

def compute_variance(
    x_query: torch.Tensor,
    x_data: torch.Tensor,
    kernel: AbstractKernel,
    var_noise: float
):
    """
    Compute the variance of the Gaussian Process at the query points.
    :math:`\sigma^2(x_*) = k(x_*, x_*) - k(x_*, X) K(X, X)^{-1} k(X, x_*)^T`

    Args:
        x_query: Query points (shape: [n_query, d]).
        x_data: Training data points (shape: [n_data, d]).
        kernel: Kernel function to compute covariance.
        var_noise: Noise level (variance).

    Returns:
        Variance at the query points (shape: [n_query, 1]).
    """
    k_xx = kernel(x_data, x_data) + var_noise * torch.eye(x_data.shape[0], device=x_data.device)  # [n_data, n_data]
    k_xq = kernel(x_data, x_query)  # [n_data, n_query]
    k_qq = kernel(x_query, x_query, diag=True)[:, torch.newaxis]  # [n_query, 1]

    alpha = solve_spd(k_xx, k_xq)  # [n_data, n_query]
    variance = k_qq - torch.sum(k_xq * alpha, dim=0, keepdim=True).T  # [n_query, 1]

    return variance