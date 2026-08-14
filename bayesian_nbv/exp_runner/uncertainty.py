from bayesian_nbv.exp_runner.base import *
from bayesian_nbv.scan.camera import cameras_on_sphere_lookat_origin, farthest_point_sampling
from bayesian_nbv.pointnet.utils import subsample_points
from numpy.random import default_rng

def sample_ray(eyes: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    eyes: (B, 3)
    """
    x1 = eyes.unsqueeze(1) # (B, 1, 3)
    x2 = -eyes.unsqueeze(1) # (B, 1, 3)
    t = torch.linspace(0, 1, num_samples, device=eyes.device).view(1, num_samples, 1) # (1, num_samples, 1)
    rays = x1 * (1 - t) + x2 * t # (B, num_samples, 3)
    return rays

def ray_uncertainty_score(transmittance: torch.Tensor) -> torch.Tensor:
    """
    transmittance: (num_samples,)
    Returns:
        score: scalar tensor
    """
    try:
        first_less_than_1 = torch.where(transmittance < 1)[0][0]
        last_greater_than_0 = torch.where(transmittance > 0.05)[0][-1]
    except IndexError:
        # print(transmittance)
        return torch.scalar_tensor(0.0, device=transmittance.device)
    return last_greater_than_0 - first_less_than_1


class RayUncertaintyRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.needs_sample = False
        self.start_indices = []
        self.cam_idx_rng = default_rng(config.camera.get('camera_idx_seed', 0))
    
    def prepare_init_camera(self, args: edict) -> tuple[torch.Tensor, torch.Tensor]:
        self.R_all, self.t_all, self.eye_all = cameras_on_sphere_lookat_origin(args.num_cameras, args.radius, device=self.device)
        start_idx = args.get('start_idx', None)
        if start_idx is None:
            # Select start indices randomly
            while True:
                start_idx = self.cam_idx_rng.integers(args.num_cameras, dtype=int)
                if self.check_init_camera(self.R_all[start_idx], self.t_all[start_idx]):
                    break
                print(f"Warning: Initial camera {start_idx} is not valid, retrying...")
            print(f"Chosen initial camera: {start_idx}")
        self.start_indices.append(start_idx)
        self.candidate_mask = torch.ones(args.num_cameras, dtype=torch.bool)
        self.candidate_mask[start_idx] = False
        self.start_idx = start_idx

        np.savez(self.result_dir / "all_cameras.npz", R=torch_to_numpy(self.R_all), t=torch_to_numpy(self.t_all), K=torch_to_numpy(self.K))
        return self.R_all[start_idx], self.t_all[start_idx]
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['selected_indices'] = [self.start_idx]
        return step_ckpt
    
    def expected_improvement(self, f_samples: torch.Tensor, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        eye_candidates = self.eye_all[self.candidate_mask]

        rays = sample_ray(eye_candidates, self.config.spsr.num_ray_samples) # (B, num_samples, 3)
        rays = rays.view(-1, 3) # (B * num_samples, 3)

        data_mean_cholesky = compute_mean(
            x_data, x_data, v_data, self.k_v, self.k_fv, self.noise_level
        ).mean()

        xi = torch.randn(self.config.spsr.num_samples, 3, self.k_v_eigenvectors.shape[0], 2, device=self.device)
        gamma = f_gamma_from_v_xi(xi, self.k_v_eigenvectors, self.k_f_eigenvalues)
        f_samples_rays = sample_pathwise_conditioning(
            rays,
            x_data,
            v_data,
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
        ) - data_mean_cholesky # (ns, B * num_samples)
        f_samples_rays = f_samples_rays.view(self.config.spsr.num_samples, -1, self.config.spsr.num_ray_samples) # (ns, B, num_samples)
        transmittance = torch.cumprod((f_samples_rays > 0).float(), dim=-1).mean(dim=0) # (B, num_samples)

        ray_scores = [ray_uncertainty_score(transmittance[i]) for i in range(transmittance.shape[0])]
        ray_scores = torch.stack(ray_scores) # (B,)

        ray_scores_all = torch.full((self.R_all.shape[0],), -float('inf'), device=self.device)
        ray_scores_all[self.candidate_mask] = ray_scores.float()
        selected_ci = torch.argmax(ray_scores_all).item()
        assert self.candidate_mask[selected_ci], "Redundant camera selected"
        self.candidate_mask[selected_ci] = False
        step_ckpt['selected_indices'].append(selected_ci)

        if self.verbose:
            msg = f"Selected camera {selected_ci} for step {self.current_step+1}"
            print(msg)
        
        return self.R_all[selected_ci], self.t_all[selected_ci], step_ckpt
        

