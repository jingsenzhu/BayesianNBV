import os
import argparse
import json

import numpy as np
import torch
from tqdm.auto import tqdm, trange

from bayesian_nbv.gp_torch.kernels import Matern32Kernel
from bayesian_nbv.spsr_torch.gp import (
    f_eigenvalues_from_v_eigenvalues,
    f_gamma_from_v_xi,
    evaluate_karhunen_loeve_sample_vectorized
)
from bayesian_nbv.utils.mesh import grid_3d_points
from bayesian_nbv.spsr_torch.utils import trunc_Zd

# Defaults below reproduce the prior samples used in the paper (precompute/l_5e-3_v_0.2):
# 25 files (000.npy - 024.npy), each holding 16 samples on a 100^3 grid.
parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=str, help='Directory to save samples')
parser.add_argument('--seed', type=int, default=32, help='Random seed')
parser.add_argument('-r', '--grid_radius', type=float, default=3.5, help='Radius of the grid for sampling')
parser.add_argument('-g', '--grid_density', type=int, default=100, help='Density of the grid for sampling')
parser.add_argument('-n', '--num_samples', type=int, default=400, help='Number of samples to generate')
parser.add_argument('-bs', '--batch_sample', type=int, default=16, help='Batch size (samples), i.e. number of samples per saved file')
parser.add_argument('-bf', '--batch_f', type=int, default=2**10, help='Batch size (f)')
parser.add_argument('-t', '--truncation_n', type=int, default=50, help='Number of eigenfunctions to use')
parser.add_argument('-l', '--lengthscale', type=float, default=5e-3, help='Kernel lengthscale')
parser.add_argument('-v', '--variance', type=float, default=0.2, help='Kernel variance')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
args = parser.parse_args()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

config = {
    "seed": args.seed,
    "grid_radius": args.grid_radius,
    "grid_density": args.grid_density,
    "truncation_n": args.truncation_n,
    "lengthscale": args.lengthscale,
    "variance": args.variance,
}
with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config, f)

assert args.num_samples % args.batch_sample == 0, "num_samples must be divisible by batch_sample"

device = torch.device(args.device)
print(f"[INFO] Using device: {device}")

x_query, _ = grid_3d_points(args.grid_density, args.grid_radius)
x_query = torch.from_numpy(x_query).float().to(device)
n_query = x_query.shape[0]

k_v = Matern32Kernel(args.lengthscale, args.variance, 3).to(device)
kl_eigenvectors = trunc_Zd(args.truncation_n).to(device)
kl_v_eigenvalues = k_v.spectral_density(kl_eigenvectors) ** 0.5
kl_f_eigenvalues = f_eigenvalues_from_v_eigenvalues(kl_eigenvectors, kl_v_eigenvalues)

generator = torch.Generator(device=device).manual_seed(args.seed)
with torch.no_grad():
    for i in tqdm(range(args.num_samples // args.batch_sample)):
        xi = torch.randn(
            (args.batch_sample, 3, kl_eigenvectors.shape[0], 2), generator=generator, device=device
        )
        gamma = f_gamma_from_v_xi(xi, kl_eigenvectors, kl_f_eigenvalues)
        n_batches = (n_query + args.batch_f - 1) // args.batch_f
        iterator = trange(n_batches, leave=False)
        f_samples = []
        for batch_i in iterator:
            start = batch_i * args.batch_f
            end = min((batch_i + 1) * args.batch_f, n_query)
            xq_batch = x_query[start:end]
            # (batch_f, batch_sample)
            f_batch = evaluate_karhunen_loeve_sample_vectorized(xq_batch, kl_eigenvectors, kl_f_eigenvalues, gamma)
            f_samples.append(f_batch)
        # (n_query, batch_sample) -> (batch_sample, n_query)
        f_samples = torch.cat(f_samples, dim=0).T.contiguous()
        np.save(f"{save_dir}/{i:03d}.npy", f_samples.cpu().numpy())
