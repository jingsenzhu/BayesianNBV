import os
import shutil
from pathlib import Path
from functools import partial
from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime
from easydict import EasyDict as edict

import numpy as np
import torch
import json
import yaml
from yaml.representer import Representer

import point_cloud_utils as pcu
import gpytoolbox

from bayesian_nbv.gp_torch.kernels import Matern32Kernel
from bayesian_nbv.pointnet.utils import subsample_points
from bayesian_nbv.spsr_torch.gp import (
    compute_mean, 
    compute_mean_batched,
    f_eigenvalues_from_v_eigenvalues,
    f_gamma_from_v_xi,
    poisson_cross_covariances_vectorized,
    sample_pathwise_conditioning
)
from bayesian_nbv.spsr_torch.utils import periodic_stationary_interpolator, trunc_Zd
from bayesian_nbv.utils.mesh import grid_3d_points, normalize_points
from bayesian_nbv.scan.scan import simulate_scan_single_camera, simulate_scan_batched
from bayesian_nbv.scan.camera import intrinsic_matrix_from_fov
from tqdm import trange

def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def numpy_to_torch(x: np.ndarray, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(x).to(device).to(dtype)

class BaseRunner(ABC):
    def __init__(self, config: edict, verbose: bool = True):
        self.config = config
        args = config.exp
        self.verbose = verbose
        self.needs_sample = True
        self.subsample_samples = True

        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed(args.seed)
        # np.random.seed(args.seed)

        # self.setup_exp_dir(args)
        # if self.exp_dir.exists():
        #     shutil.rmtree(self.exp_dir)
        # self.exp_dir.mkdir(parents=True)
        # self.result_dir = self.exp_dir / self.exp_name
        # self.result_dir.mkdir(parents=True)
        self.precompute_dir = Path(args.precompute_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_steps = args.num_steps

        if (self.precompute_dir / 'config.json').exists():
            with open(self.precompute_dir / 'config.json', 'r') as f:
                spsr_cfg = json.load(f)
            self.config.spsr.update(spsr_cfg)
        
        spsr_cfg = self.config.spsr
        sample0 = np.load(self.precompute_dir / "000.npy")
        n_samples, n_cube = sample0.shape
        if spsr_cfg.get('num_samples', None) is None:
            spsr_cfg.num_samples = n_samples
            self.n_loads = 1
        else:
            assert spsr_cfg.num_samples % n_samples == 0, "num_samples must be divisible by n_samples"
            self.n_loads = spsr_cfg.num_samples // n_samples
        assert n_cube == spsr_cfg.grid_density**3, "Number of cube points must be equal to grid_density^3"

        
        self.grid_density = self.config.spsr.grid_density
        self.noise_level = self.config.spsr.noise_level
        self.setup_spsr(self.config.spsr)

        self.max_points = self.config.points.max_points
        self.subsample_points = self.config.points.subsample_points
        self.scan_batch_size = self.config.points.get('scan_batch_size', None)
        assert self.scan_batch_size is None or self.scan_batch_size > 0, "scan_batch_size must be None or positive"
        self.spsr_batch_size = self.config.spsr.batch_size
        self.acquisition = args.acquisition
        
        self.setup()
    
    def set_experiment(self, exp_args: edict):
        torch.manual_seed(exp_args.seed)
        torch.cuda.manual_seed(exp_args.seed)
        np.random.seed(exp_args.seed)
        if self.exp_dir.exists():
            shutil.rmtree(self.exp_dir)
        self.exp_dir.mkdir(parents=True)
        self.result_dir.mkdir(parents=True)
        self.config.exp.update(exp_args)

        yaml.SafeDumper.add_representer(edict, Representer.represent_dict)
        with open(self.exp_dir / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        
    @abstractmethod
    def check_existing(self) -> bool:
        pass

    @abstractmethod
    def setup_exp_dir(self, args: edict):
        self.exp_dir = Path('') # to be implemented in subclass
        self.exp_name = '' # to be implemented in subclass
    
    def setup_spsr(self, args: edict):
        lengthscale = args.lengthscale
        variance = args.variance
        truncation_n = args.truncation_n
        amortization_density = args.amortization_density

        self.k_v = Matern32Kernel(lengthscale, variance, 3).to(self.device)

        self.k_v_eigenvectors = trunc_Zd(truncation_n).to(self.device)
        self.k_v_eigenvalues = self.k_v.spectral_density(self.k_v_eigenvectors) ** 0.5

        k_fv_expensive = partial(
            poisson_cross_covariances_vectorized,
            eigenvectors=self.k_v_eigenvectors,
            eigenvalues=self.k_v_eigenvalues,
        )
        self.k_fv = periodic_stationary_interpolator(
            k_fv_expensive, 3, amortization_density, device=self.device, exponent=5, verbose=True, bs=512
        )

        self.k_f_eigenvalues = f_eigenvalues_from_v_eigenvalues(self.k_v_eigenvectors, self.k_v_eigenvalues)
        self.k_f_variance = (self.k_f_eigenvalues**2).sum()

        x_grid, _ = grid_3d_points(self.grid_density, args.grid_radius)
        self.x_grid = numpy_to_torch(x_grid, self.device)

    @abstractmethod
    def setup(self):
        pass

    def preprocess_mesh(self, mesh_file: str, v: np.ndarray, f: np.ndarray):
        return v, f
    
    @abstractmethod
    def prepare_init_camera(self, args: edict) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod
    def check_init_camera(self, R: torch.Tensor, t: torch.Tensor) -> bool:
        pass
    
    def load_mesh_and_init_scan(self, args: edict, mesh_file: str):
        if not os.path.exists(mesh_file):
            mesh_file = mesh_file.replace('.off', '.ply')
        v, f = pcu.load_mesh_vf(mesh_file)
        v, f = self.preprocess_mesh(mesh_file, v, f)
        pcu.save_mesh_vf(str(self.result_dir / "mesh.ply"), v, f)
        self.V_torch = numpy_to_torch(v, self.device)
        self.F_torch = numpy_to_torch(f, self.device, dtype=torch.long)

        self.K = intrinsic_matrix_from_fov(args.fov, args.resolution[1], args.resolution[0], device=self.device)
        R_init, t_init = self.prepare_init_camera(self.config.camera)
        points, normals, _, mask = simulate_scan_single_camera(
            self.V_torch, self.F_torch, None, 
            R_init, t_init, self.K, 
            args.resolution[0], args.resolution[1], device=self.device, 
            flip_back_normals=True
        )

        points = points[mask]
        normals = normals[mask]
        points = points + torch.randn_like(points) * self.noise_level
        normals = normals + torch.randn_like(normals) * self.noise_level
        return points, normals, torch_to_numpy(R_init), torch_to_numpy(t_init)

    @abstractmethod
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        return kwargs
    
    @abstractmethod
    def step_utility(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        pass
    
    def post_reconstruction(self, V_step: np.ndarray, F_step: np.ndarray, step_ckpt: dict[str, Any]):
        return step_ckpt

    def step_reconstruction(self, save_name: str, x_data: torch.Tensor, v_data: torch.Tensor, data_mean_cholesky: torch.Tensor, step_ckpt: dict[str, Any]):
        grid_mean_step = compute_mean_batched(
            self.x_grid, x_data, v_data, self.k_v, self.k_fv, self.noise_level, self.spsr_batch_size, verbose=False
        )
        f_step = data_mean_cholesky - grid_mean_step
        V_step, F_step = gpytoolbox.marching_cubes(
            torch_to_numpy(f_step), torch_to_numpy(self.x_grid), 
            self.grid_density, self.grid_density, self.grid_density, 
            0.0
        )
        gpytoolbox.write_mesh(f"{self.result_dir}/{save_name}.obj", V_step, F_step)
        step_ckpt = self.post_reconstruction(V_step, F_step, step_ckpt)
        return step_ckpt
    
    @abstractmethod
    def expected_improvement(self, f_samples: torch.Tensor, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        pass

    @abstractmethod
    def expected_acquisition(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def save_results(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        pass

    def run(self, exp_args: edict, override_existing: bool = False):
        self.setup_exp_dir(exp_args)
        if not override_existing and self.check_existing():
            print(f"[INFO] Skipping existing experiment: {self.exp_name}")
            return
        self.set_experiment(exp_args)
        x_data, v_data, R_init, t_init = self.load_mesh_and_init_scan(self.config.scan, exp_args.mesh_file)
        self.x_data_current = x_data
        self.v_data_current = v_data

        init_scan = torch_to_numpy(torch.cat([x_data, v_data], dim=-1)) # (N, 6)
        step_ckpt = self.init_step_checkpoint(step_scans=[init_scan], Rs_selected=[R_init], ts_selected=[t_init])
        self.current_step = 0

        for i in trange(self.num_steps - 1, disable=not self.verbose):
            # if self.max_points is not None and x_data.shape[0] > self.max_points:
            #     idx = np.random.choice(np.arange(x_data.shape[0]), size=(self.max_points,), replace=False)
            #     x_data_ = x_data[idx]
            #     v_data_ = v_data[idx]
            # else:
            #     x_data_ = x_data
            #     v_data_ = v_data
            step_ckpt = self.step_utility(x_data, v_data, step_ckpt)

            # Subsample points for SPSR to avoid memory issues
            x_data_input, v_data_input = subsample_points(x_data, v_data, self.max_points, dup_pts=False, random_start_point=True)

            data_mean_cholesky = compute_mean(
                x_data_input, x_data_input, v_data_input, self.k_v, self.k_fv, self.noise_level
            ).mean()

            if self.config.exp.step_reconstruction:
                self.step_reconstruction(f"step_{i:03d}_reconstruction", x_data_input, v_data_input, data_mean_cholesky)
            
            if self.needs_sample:
                f_prior = []
                for j in range(self.n_loads):
                    f_prior_j = numpy_to_torch(np.load(self.precompute_dir / f"{i * self.n_loads + j:03d}.npy"), self.device)
                    f_prior.append(f_prior_j)
                f_prior = torch.cat(f_prior, dim=0) # (args.num_samples, n_cube)

                xi = torch.randn(self.config.spsr.num_samples, 3, self.k_v_eigenvectors.shape[0], 2, device=self.device)
                gamma = f_gamma_from_v_xi(xi, self.k_v_eigenvectors, self.k_f_eigenvalues)

                f_samples = sample_pathwise_conditioning(
                    self.x_grid,
                    x_data_input,
                    v_data_input,
                    self.k_v,
                    self.k_fv,
                    self.noise_level,
                    xi,
                    gamma,
                    self.k_v_eigenvectors,
                    self.k_v_eigenvalues,
                    self.k_v_eigenvectors,
                    self.k_f_eigenvalues,
                    2**8,
                    2**8,
                    2**10,
                    verbose=False,
                    precompute_f=f_prior,
                )

                R_selected, t_selected, step_ckpt = self.expected_improvement(data_mean_cholesky - f_samples, x_data, v_data, step_ckpt)
            else:
                R_selected, t_selected, step_ckpt = self.expected_improvement(None, x_data_input, v_data_input, step_ckpt)

            points, normals, _, mask = simulate_scan_single_camera(
                self.V_torch, self.F_torch, None, 
                R_selected, t_selected, self.K, 
                self.config.scan.resolution[0], self.config.scan.resolution[1], device=self.device, 
                flip_back_normals=True
            )  # (H,W,3), (H,W,3), (H,W), (H,W)
            points = points[mask]
            normals = normals[mask]
            points = points + torch.randn_like(points) * self.noise_level
            normals = normals + torch.randn_like(normals) * self.noise_level

            step_ckpt['step_scans'].append(np.concatenate([
                torch_to_numpy(points), torch_to_numpy(normals)
            ], axis=-1))
            step_ckpt['Rs_selected'].append(torch_to_numpy(R_selected))
            step_ckpt['ts_selected'].append(torch_to_numpy(t_selected))

            x_data = torch.cat([x_data, points], dim=0)
            v_data = torch.cat([v_data, normals], dim=0)
            self.x_data_current = x_data
            self.v_data_current = v_data
            self.current_step += 1

        step_ckpt = self.step_utility(x_data, v_data, step_ckpt)

        x_data_input, v_data_input = subsample_points(x_data, v_data, self.max_points, dup_pts=False, random_start_point=True)
        data_mean_cholesky = compute_mean(
            x_data_input, x_data_input, v_data_input, self.k_v, self.k_fv, self.noise_level
        ).mean()
        step_ckpt = self.step_reconstruction(f"final_reconstruction", x_data_input, v_data_input, data_mean_cholesky, step_ckpt)

        self.save_results(x_data, v_data, step_ckpt)


    def verbose_msg(self, msg, step_ckpt, expected_acqisition):
        return msg




        