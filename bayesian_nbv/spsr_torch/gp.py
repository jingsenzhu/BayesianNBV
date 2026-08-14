import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.auto import tqdm, trange
from bayesian_nbv.gp_torch.gp import solve_spd

@torch.compile
def poisson_cross_covariances_vectorized(x_diff, eigenvectors, eigenvalues):
    """
    Compute the cross-covariance between f and v of Poisson Surface Reconstruction
    given the Karhunen-Loeve expansion of the kernel on v

    x_diff: (m, d)
    evecs: (n, d)
    evals: (n,)

    returns: (m, d)
    """
    evec_norm_sq = (eigenvectors**2).sum(-1)  # (n,)
    evec_norm_sq = torch.where(evec_norm_sq > 0, evec_norm_sq, 1)  # (n,)
    dot = x_diff @ eigenvectors.T  # (m, d) @ (d, n) = (m, n)
    sin_ = torch.sin(dot)
    coef = (eigenvalues.unsqueeze(0))**2 * sin_ / evec_norm_sq.unsqueeze(0)  # (m, n)
    result = coef @ eigenvectors  # (m, n) @ (n, d) = (m, d)
    return result

def f_eigenvalues_from_v_eigenvalues(eigenvectors, eigenvalues):
    """
    Compute the Karhunen-Loeve expansion's eigenvalues of the kernel on f given the
    Mercer expansion of the kernel on v

    eigenvectors: (n, d)
    eigenvalues: (n,)

    returns: (n,)
    """
    factor = ((eigenvectors * eigenvalues[:, None]) ** 2).sum(-1) ** 0.5
    evec_norms = (eigenvectors**2).sum(-1)
    return torch.where(evec_norms > 0, factor / evec_norms, 0)

def f_gamma_from_v_xi(xi, eigenvectors, eigenvalues):
    """
    Compute the random variables in the Karhunen-Loeve expansion of samples of f
    given the Karhunen-Loeve expansion of samples of v

    xi: (m, d, n, 2)
    eigenvectors: (n, d)
    eigenvalues: (n,)

    returns: (m, n, 2)
    """
    factor = (eigenvectors**2 * eigenvalues[:, None] ** 2).sum(-1) ** 0.5
    gamma = (
        eigenvectors.T[..., None]
        * eigenvalues[..., None]
        * torch.flip(xi, [-1])
        * torch.tensor([-1, 1], dtype=xi.dtype, device=xi.device)
    ).sum(1) / torch.where(factor > 0, factor, 1)[..., None]

    return gamma

@torch.compile
def compute_mean(
    x_query,
    x_data,
    v_data,
    kernel_v,
    kernel_fv,
    sigma,
):
    """
    Compute the mean of f at x_query conditioned on the v observations of v_data
    at x_data

    x_query: (nq, d)
    x_data: (nd, d)
    v_data: (nd, d)

    returns: (nq,)
    """
    kv_dd = (
        kernel_v(x_data, x_data)
        + torch.eye(x_data.shape[0], device=x_data.device) * (sigma**2)
    )
    alpha = solve_spd(kv_dd, v_data) # (nd, d)

    kfv = torch.vmap(kernel_fv, (0, None))(x_query, x_data)  # (nq, nd, d)

    return torch.einsum('ijk,jk->i', kfv, alpha)  # (nq,)

@torch.compile
def mean_batched(xq_batch, x_data, alpha, v_kernel_fv):
    kfv_batch = v_kernel_fv(xq_batch, x_data)
    return torch.einsum('ijk,jk->i', kfv_batch, alpha)

def compute_mean_batched(
    x_query,
    x_data,
    v_data,
    kernel_v,
    kernel_fv,
    sigma,
    batch_size,
    verbose=False
):
    """
    Compute the mean of f at x_query conditioned on the v observations of v_data
    at x_data

    x_query: (nq, d)
    x_data: (nd, d)
    v_data: (nd, d)
    batch_size: int

    returns: (nq,)
    """
    kv_dd = (
        kernel_v(x_data, x_data)
        + torch.eye(x_data.shape[0], device=x_data.device) * (sigma**2)
    )
    alpha = solve_spd(kv_dd, v_data) # (nd, d)

    n_batches = (x_query.shape[0] + batch_size - 1) // batch_size
    mean_batches = []
    v_kernel_fv = torch.vmap(kernel_fv, (0, None))
    for i in trange(n_batches, disable=not verbose, desc="Computing kfv"):
        start = i * batch_size
        end = min((i + 1) * batch_size, x_query.shape[0])
        xq_batch = x_query[start:end]
        kfv_batch = v_kernel_fv(xq_batch, x_data)
        mean_batch = torch.einsum('ijk,jk->i', kfv_batch, alpha)
        # mean_batches.append(mean_batched(xq_batch, x_data, alpha, v_kernel_fv))
        mean_batches.append(mean_batch)
    
    mean = torch.cat(mean_batches, dim=0)  # (nq,)   
    return mean  # (nq,)

@torch.compile
def compute_variance(
    x_query,
    x_data,
    kernel_v,
    kernel_fv,
    var_f,
    sigma
):
    """
    Compute the variance of f at x_query conditioned on the v observations of v_data
    at x_data

    x_query: (nq, d)
    x_data: (nd, d)

    returns: (nq, nq)
    """
    kv_dd = (
        kernel_v(x_data, x_data)
        + torch.eye(x_data.shape[0], device=x_data.device) * (sigma**2)
    )
    kv_dd_b = torch.broadcast_to(kv_dd.unsqueeze(0), (x_query.shape[1], x_data.shape[0], x_data.shape[0]))  # [d, nd, nd]

    kfv = torch.vmap(kernel_fv, (0, None))(x_query, x_data)  # (nq, nd, d)
    kvf = kfv.T  # (d, nd, nq)
    alpha = solve_spd(kv_dd_b, kvf)  # (d, nd, nq)
    return var_f - torch.einsum("ijk,kji->i", kfv, alpha)  # (nq, nd, d) @ (d, nd, nq) -> (nq,)


def evaluate_karhunen_loeve_sample(x, eigenvectors, eigenvalues, xi):
    """
    Evaluate the Karhunen-Loeve expansion of samples

    x: (d,)
    eigenvectors: (n, d)
    eigenvalues: (n,)
    xi: (n, 2)

    returns: (,)
    """
    dot = torch.einsum("id,d->i", eigenvectors, x) # (n,)

    cos_term = xi[:, 0] * torch.cos(dot)
    sin_term = xi[:, 1] * torch.sin(dot)

    return (eigenvalues * (cos_term + sin_term)).sum()


@torch.compile
def evaluate_karhunen_loeve_sample_vectorized(x, eigenvectors, eigenvalues, xi):
    """
    Evaluate the Karhunen-Loeve expansion for batched x and batched xi
    
    x: (nx, d) - batch of input points
    eigenvectors: (n, d)
    eigenvalues: (n,)
    xi: (..., n, 2) - arbitrary number of batch dimensions
    
    returns: (nx, ...) - combines x batch dimension with xi batch dimensions
    """
    # Compute dot products for all x vectors
    # x: (nx, d), eigenvectors: (n, d) -> dots: (nx, n)
    dots = torch.einsum("id,jd->ij", x, eigenvectors)  # (nx, n)
    
    # Get the batch shape from xi (all dimensions except the last two)
    batch_shape = xi.shape[:-2]
    n = xi.shape[-2]
    nx = x.shape[0]
    
    # Reshape dots for broadcasting with xi
    # dots: (nx, n) -> (nx, 1, 1, ..., 1, n) with len(batch_shape) ones
    reshape_dims = (nx,) + (1,) * len(batch_shape) + (n,)
    dots = dots.view(reshape_dims)
    
    # Reshape eigenvalues for broadcasting
    # eigenvalues: (n,) -> (1, 1, ..., 1, n) with (1 + len(batch_shape)) ones
    eigenvalues_reshape = (1,) + (1,) * len(batch_shape) + (n,)
    eigenvalues = eigenvalues.view(eigenvalues_reshape)
    
    # Reshape xi for broadcasting with x dimension
    # xi: (..., n, 2) -> (1, ..., n, 2)
    xi_reshape = (1,) + xi.shape
    xi_expanded = xi.view(xi_reshape)
    
    # Extract cos and sin coefficients
    # xi_expanded[..., 0]: (1, ..., n), xi_expanded[..., 1]: (1, ..., n)
    xi_cos = xi_expanded[..., 0]  # (1, ..., n)
    xi_sin = xi_expanded[..., 1]  # (1, ..., n)
    
    # Compute cos and sin terms with broadcasting
    # dots: (nx, 1, ..., 1, n) broadcasts with xi_cos/xi_sin: (1, ..., n)
    # Result: (nx, ..., n)
    cos_term = xi_cos * torch.cos(dots)  # (nx, ..., n)
    sin_term = xi_sin * torch.sin(dots)  # (nx, ..., n)
    
    # Compute final result and sum over the last dimension
    # eigenvalues broadcasts to (nx, ..., n)
    result = eigenvalues * (cos_term + sin_term)  # (nx, ..., n)
    return result.sum(dim=-1)  # (nx, ...)

@torch.compile
def kfv_update_batch(xq_batch, x_data, kernel_fv, alpha):
    """
    xq_batch: (batch, d)
    x_data: (Nd, d)
    alpha: (ns, Nd, 3)
    """
    # xq_batch: (batch, d), x_data: (Nd, d), alpha: (ns, Nd, 3)
    # kernel_fv(xq_batch[:, None, :], x_data[None, :, :]) -> (batch, Nd, 3)
    Nb = xq_batch.shape[0]
    Nd = x_data.shape[0]
    d = xq_batch.shape[1]
    xq_batch_exp = xq_batch.unsqueeze(1).expand(Nb, Nd, -1).reshape(-1, d)
    x_data_exp = x_data.unsqueeze(0).expand(Nb, Nd, -1).reshape(-1, d)
    kfv_matrix = kernel_fv(xq_batch_exp, x_data_exp)  # (Nb*Nd, 3)
    kfv_matrix = kfv_matrix.view(Nb, Nd, 3)
    # alpha: (ns, Nd, 3)
    # We want (ns, batch)
    # print(f"kfv_matrix shape: {kfv_matrix.shape}, alpha shape: {alpha.shape}")
    kfv_matrix_exp = kfv_matrix.unsqueeze(0)  # (1, batch, Nd, 3)
    alpha_exp = alpha.unsqueeze(1)  # (ns, 1, Nd, 3)
    # print(f"kfv_matrix_exp shape: {kfv_matrix_exp.shape}, alpha_exp shape: {alpha_exp.shape}")
    return (kfv_matrix_exp * alpha_exp).sum(dim=(2, 3))  # (ns, batch)

def sample_pathwise_conditioning(
    x_query,
    x_data,
    v_data,
    kernel_v,
    kernel_fv,
    sigma,
    xi,
    gamma,
    v_eigenvectors,
    v_eigenvalues,
    f_eigenvectors,
    f_eigenvalues,
    bs_v,
    bs_kfv,
    bs_f,
    sdd_params=None,
    verbose=False,
    precompute_f=None
    # use_jit=True,
):
    """
    Compute samples of f at x_query conditioned on the v observations of v_data
    at x_data using pathwise conditioning

    x_query: (nq, d)
    x_data: (nd, d)
    xi: (ns, 3, nv, 2)
    gamma: (ns, nv, 2)
    v_eigenvectors: (nv, d)
    v_eigenvalues: (nv,)
    f_eigenvectors: (nf,)
    f_eigenvalues: (nf,)

    returns: (ns, nq)
    """

    n_batchs = (x_data.shape[0] + bs_v - 1) // bs_v
    v_prior_batches = []
    for i in trange(n_batchs, disable=not verbose, desc="Computing prior V samples"):
        start = i * bs_v
        end = min((i + 1) * bs_v, x_data.shape[0])
        x_batch = x_data[start:end]
        v_prior_batches.append(evaluate_karhunen_loeve_sample_vectorized(x_batch, v_eigenvectors, v_eigenvalues, xi))
    v_prior = torch.cat(v_prior_batches, dim=0)  # (Nd, ns, d)
    v_prior = v_prior.transpose(0, 1)  # (ns, Nd, d)
    v_residual = v_data.unsqueeze(0) - v_prior  # (ns, Nd, d)

    # if not sdd_params:
    #     kv_dd = (
    #         kernel_v(x_data, x_data)
    #         + torch.eye(x_data.shape[0], device=x_data.device) * (sigma**2)
    #     )
    #     solver = partial(solve_spd, A=kv_dd)
    # else:
    #     raise NotImplementedError("SDD solver not implemented")
        # solver = partial(solver_sdd, x_data=x_data, kernel_v=kernel_v, sigma=sigma, **sdd_params)
    # alpha = torch.vmap(solver)(v_residual)
    kv_dd = (
        kernel_v(x_data, x_data)
        + torch.eye(x_data.shape[0], dtype=x_data.dtype, device=x_data.device) * (sigma**2)
    )
    alpha = torch.vmap(solve_spd, (None, 0))(kv_dd, v_residual)  # (ns, Nd, d)

    # @torch.compile
    # def kfv_update_batch(xq_batch):
    #     # xq_batch: (batch, d)
    #     # kernel_fv(xq_batch[:, None, :], x_data[None, :, :]) -> (batch, Nd, 3)
    #     kfv_matrix = kernel_fv(xq_batch[:, None, :], x_data[None, :, :])  # (batch, Nd, 3)
    #     # alpha: (ns, Nd, 3)
    #     # We want (ns, batch)
    #     kfv_matrix_exp = kfv_matrix.unsqueeze(0)           # (1, batch, Nd, 3)
    #     alpha_exp = alpha.unsqueeze(1)                  # (ns, 1, Nd, 3)
    #     return (kfv_matrix_exp * alpha_exp).sum(dim=(2, 3))  # (ns, batch)

    n_query = x_query.shape[0]
    f_update_batches = []
    n_batches = (n_query + bs_kfv - 1) // bs_kfv
    for batch_i in trange(n_batches, disable=not verbose, desc="Computing kfv update"):
        start = batch_i * bs_kfv
        end = min((batch_i + 1) * bs_kfv, n_query)
        xq_batch = x_query[start:end]
        # f_update_batches.append(kfv_update_batch(xq_batch))
        f_update_batches.append(kfv_update_batch(xq_batch, x_data, kernel_fv, alpha))
    f_update = torch.cat(f_update_batches, dim=1)  # (ns, Nq)

    if precompute_f is None:
        n_query = x_query.shape[0]
        n_batches = (n_query + bs_f - 1) // bs_f
        f_prior_batches = []
        for batch_i in trange(n_batches, disable=not verbose, desc="Computing f prior samples"):
            start = batch_i * bs_f
            end = min((batch_i + 1) * bs_f, n_query)
            xq_batch = x_query[start:end]
            # (ns, batch)
            f_prior_batch = evaluate_karhunen_loeve_sample_vectorized(
                xq_batch, f_eigenvectors, f_eigenvalues, gamma
            )
            f_prior_batches.append(f_prior_batch)
        f_prior = torch.cat(f_prior_batches, dim=0)  # (Nq, ns)
        f_prior = f_prior.transpose(0, 1)  # (ns, Nq)
    else:
        f_prior = precompute_f

    return f_prior + f_update