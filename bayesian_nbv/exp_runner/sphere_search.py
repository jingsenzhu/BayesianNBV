from bayesian_nbv.exp_runner.base import *
from bayesian_nbv.scan.camera import cameras_on_sphere_lookat_origin
from bayesian_nbv.pointnet.utils import subsample_points
from numpy.random import default_rng

class SphereSearchRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.start_indices = []
        self.cam_idx_rng = default_rng(config.camera.get('camera_idx_seed', 0))

    def prepare_init_camera(self, args: edict) -> tuple[torch.Tensor, torch.Tensor]:
        self.R_all, self.t_all, _ = cameras_on_sphere_lookat_origin(args.num_cameras, args.radius, device=self.device)
        self.candidate_mask = torch.ones(args.num_cameras, dtype=torch.bool)
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
        self.candidate_mask[start_idx] = False
        np.savez(self.result_dir / "all_cameras.npz", R=torch_to_numpy(self.R_all), t=torch_to_numpy(self.t_all), K=torch_to_numpy(self.K))
        return self.R_all[start_idx], self.t_all[start_idx]
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['selected_indices'] = [self.start_indices[-1]]
        return step_ckpt
    
    def expected_improvement(self, f_samples: torch.Tensor, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        R_candidates = self.R_all[self.candidate_mask]
        t_candidates = self.t_all[self.candidate_mask]
        expected_acquisitions = []
        for j in range(f_samples.shape[0]):
            f_sample = f_samples[j]
            V_j, F_j = gpytoolbox.marching_cubes(
                torch_to_numpy(f_sample), torch_to_numpy(self.x_grid), 
                self.grid_density, self.grid_density, self.grid_density, 
                0.0
            )
            V_j_torch = numpy_to_torch(V_j, self.device)
            F_j_torch = numpy_to_torch(F_j, self.device)
            if self.scan_batch_size is None:
                points, normals, _, masks = simulate_scan_batched(
                    V_j_torch, F_j_torch, None,
                    R_candidates, t_candidates, self.K,
                    self.config.scan.resolution[0], self.config.scan.resolution[1], device=self.device,
                    flip_back_normals=True
                )
            else:
                n_scan_batches = (R_candidates.shape[0] + self.scan_batch_size - 1) // self.scan_batch_size
                points_list = []
                normals_list = []
                masks_list = []
                for bi in range(n_scan_batches):
                    start = bi * self.scan_batch_size
                    end = min(start + self.scan_batch_size, R_candidates.shape[0])
                    R_batch = R_candidates[start:end]
                    t_batch = t_candidates[start:end]
                    points_b, normals_b, _, masks_b = simulate_scan_batched(
                        V_j_torch, F_j_torch, None,
                        R_batch, t_batch, self.K,
                        self.config.scan.resolution[0], self.config.scan.resolution[1], device=self.device,
                        flip_back_normals=True
                    )
                    points_list.append(points_b)
                    normals_list.append(normals_b)
                    masks_list.append(masks_b)
                points = torch.cat(points_list, dim=0)
                normals = torch.cat(normals_list, dim=0)
                masks = torch.cat(masks_list, dim=0)

            next_x_data_sims = []
            next_v_data_sims = []

            expected_acquisition = torch.full((masks.shape[0],), -float('inf'), device=self.device)
            scan_mask = torch.ones_like(expected_acquisition, dtype=torch.bool)

            for ci in range(masks.shape[0]):
                mask = masks[ci]
                pts_num = mask.sum()
                if pts_num == 0:
                    print(f"Warning: No points found for camera {ci} in sample {j}")
                    scan_mask[ci] = False
                    continue
                scanned_points = points[ci][mask]
                scanned_normals = normals[ci][mask]
                scanned_points += torch.randn_like(scanned_points) * self.noise_level
                scanned_normals += torch.randn_like(scanned_normals) * self.noise_level
                next_x_data_sim = torch.cat([scanned_points, x_data], dim=0)
                next_v_data_sim = torch.cat([scanned_normals, v_data], dim=0)

                if self.subsample_samples:
                    next_x_data_sim, next_v_data_sim = subsample_points(next_x_data_sim, next_v_data_sim, self.subsample_points, random_start_point=True)
                next_x_data_sims.append(next_x_data_sim)
                next_v_data_sims.append(next_v_data_sim)

            if self.subsample_samples:
                next_x_data_sims = torch.stack(next_x_data_sims, dim=0) # (C,N,3)
                next_v_data_sims = torch.stack(next_v_data_sims, dim=0) # (C,N,3)

            expected_acquisition[scan_mask] = self.expected_acquisition(next_x_data_sims, next_v_data_sims, step_ckpt)
            expected_acquisitions.append(expected_acquisition)
        
        expected_acquisitions = torch.stack(expected_acquisitions, dim=0) # (S,C)
        expected_acquisitions_mean = expected_acquisitions.mean(dim=0) # (C,)
        acquisitions_all = torch.full((self.R_all.shape[0],), -float('inf'), device=self.device)
        acquisitions_all[self.candidate_mask] = expected_acquisitions_mean
        selected_ci = torch.argmax(acquisitions_all).item()
        assert self.candidate_mask[selected_ci], "Redundant camera selected"
        self.candidate_mask[selected_ci] = False
        step_ckpt['selected_indices'].append(selected_ci)

        if self.verbose:
            msg = f"Selected camera {selected_ci} for step {self.current_step+1}"
            msg = self.verbose_msg(msg, step_ckpt, acquisitions_all[selected_ci].item())
            print(msg)

        return self.R_all[selected_ci], self.t_all[selected_ci], step_ckpt
    
    def save_results(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        super().save_results(x_data, v_data, step_ckpt)

        # Save selected indices
        selected_indices = np.array(step_ckpt['selected_indices'], dtype=int)
        np.savetxt(self.result_dir / "selected_indices.txt", selected_indices, fmt='%d')