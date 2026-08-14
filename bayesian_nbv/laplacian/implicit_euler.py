import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

def solve_heat_with_source(L: sp.csr_matrix, M: sp.csr_matrix, u_init: np.ndarray, f: np.ndarray, T: float, n_steps: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Implicit Euler solve for the heat equation with source:
        u_t = Δu + f
    on a triangle mesh.

    Semi-discrete form (assuming L ≈ stiffness matrix, i.e. -Δ):
        M u_t = -L u + M f

    Discrete implicit Euler step:
        (M + dt L) u^{k+1} = M u^k + dt M f

    Parameters
    ----------
    L : (n, n) sparse matrix
        Cotangent Laplacian *stiffness* matrix (SPD, approximates -Δ).
    M : (n, n) sparse matrix
        Mass matrix (lumped or full, symmetric positive definite).
    u_init : (n,) array_like
        Initial per-vertex temperature.
    f : (n,) array_like
        Per-vertex heat source (assumed constant in time).
    T : float
        Final time.
    n_steps : int
        Number of implicit Euler steps to take (dt = T / n_steps).

    Returns
    -------
    u_T : (n,) ndarray
        Temperature at time t = T.
    """
    u = np.asarray(u_init).ravel()
    f = np.asarray(f).ravel()

    if T == 0.0 or n_steps <= 0:
        return u.copy()

    dt = T / float(n_steps)

    # Precompute matrices and factorization
    A = M + dt * L               # SPD system matrix
    solve_A = factorized(A.tocsc())

    Mf = M @ f                   # source term in mass-weighted form (constant if f is constant)

    u_history = [u.copy()]
    for _ in range(n_steps):
        b = M @ u + dt * Mf      # right-hand side: M u^k + dt M f
        u = solve_A(b)
        u_history.append(u.copy())

    return u, u_history