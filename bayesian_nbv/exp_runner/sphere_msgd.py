import torch
import torch.nn as nn
import torch.optim as optim
import gc
from bayesian_nbv.exp_runner.base import *
from bayesian_nbv.scan.camera import cameras_on_sphere_lookat_origin, cameras_looking_at_origin, fibonacci_sphere
from bayesian_nbv.pointnet.utils import subsample_points
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

class SphereGradientDescentRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.multi_start_batch_size = self.config.camera.get('multi_start_batch_size', None)
        assert self.multi_start_batch_size is None or self.multi_start_batch_size > 0, "multi_start_batch_size must be None or positive"
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['camera_opt_history'] = [[]] * (self.num_steps - 1)
        return step_ckpt

    def prepare_init_camera(self, args: edict) -> tuple[torch.Tensor, torch.Tensor]:
        R_all, t_all, _ = cameras_on_sphere_lookat_origin(args.num_cameras, args.radius, device=self.device)
        return R_all[args.start_idx], t_all[args.start_idx]
    
    @torch.compile
    def simulate_scan(self, V_j: torch.Tensor, F_j: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.scan_batch_size is None or self.scan_batch_size >= R.shape[0]:
            points, normals, _, masks = simulate_scan_batched(
                V_j, F_j, None,
                R, t, self.K,
                self.config.scan.resolution[0], self.config.scan.resolution[1], device=self.device,
                flip_back_normals=True
            )
        else:
            n_scan_batches = (R.shape[0] + self.scan_batch_size - 1) // self.scan_batch_size
            points_list = []
            normals_list = []
            masks_list = []
            for bi in range(n_scan_batches):
                start = bi * self.scan_batch_size
                end = min(start + self.scan_batch_size, R.shape[0])
                R_batch = R[start:end]
                t_batch = t[start:end]
                points_b, normals_b, _, masks_b = simulate_scan_batched(
                    V_j, F_j, None,
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
        return points, normals, masks
    
    @torch.compile
    def simulate_next_data(self, x_data: torch.Tensor, v_data: torch.Tensor, points: torch.Tensor, normals: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_x_data_sims = []
        next_v_data_sims = []
        # scan_mask = torch.ones(masks.shape[0], dtype=torch.bool)
        for ci in range(masks.shape[0]):
            mask = masks[ci]
            pts_num = mask.sum()
            if pts_num == 0:
                # print(f"Warning: No points found for camera {ci}")
                # scan_mask[ci] = False
                raise RuntimeError(f"No points found for camera {ci}")
            scanned_points = points[ci][mask]
            scanned_normals = normals[ci][mask]
            scanned_points += torch.randn_like(scanned_points) * self.noise_level
            scanned_normals += torch.randn_like(scanned_normals) * self.noise_level
            next_x_data_sim = torch.cat([scanned_points, x_data], dim=0)
            next_v_data_sim = torch.cat([scanned_normals, v_data], dim=0)
            next_x_data_sim, next_v_data_sim = subsample_points(next_x_data_sim, next_v_data_sim, self.subsample_points)
            next_x_data_sims.append(next_x_data_sim)
            next_v_data_sims.append(next_v_data_sim)
        next_x_data_sims = torch.stack(next_x_data_sims, dim=0) # (C,N,3)
        next_v_data_sims = torch.stack(next_v_data_sims, dim=0) # (C,N,3)
        return next_x_data_sims, next_v_data_sims #, scan_mask
    
    def scan_and_simulate(self, V: torch.Tensor, F: torch.Tensor, R: torch.Tensor, t: torch.Tensor, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]) -> torch.Tensor:
        """
        V, F: mesh parameters
        R, t: batch of cameras (B,3,3) and (B,3)
        x_data, v_data: existing data points (N,3) and (N,3)
        """
        points, normals, masks = self.simulate_scan(V, F, R, t)
        assert points.requires_grad
        next_x_data_sims, next_v_data_sims = self.simulate_next_data(x_data, v_data, points, normals, masks) # (C,N,3)
        assert next_x_data_sims.requires_grad
        expected_acquisition = self.expected_acquisition(next_x_data_sims, next_v_data_sims, step_ckpt, requires_grad=True)
        return expected_acquisition


    @torch.enable_grad()
    def expected_improvement(self, f_samples: torch.Tensor, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # R_starts, t_starts, _ = cameras_on_sphere_lookat_origin(self.config.camera.num_starts, self.config.camera.radius, device=self.device, random_oriented=True)
        # R_params = nn.Parameter(matrix_to_rotation_6d(R_starts))
        # t_params = nn.Parameter(t_starts)
        eye_params = nn.Parameter(fibonacci_sphere(self.config.camera.num_starts, radius=self.config.camera.radius, device=self.device, random_oriented=True))
        # optimizer = optim.Adam([R_params, t_params], lr=self.config.camera.learning_rate)

        optimizer = getattr(optim, self.config.camera.optimizer)([eye_params], lr=self.config.camera.learning_rate, **self.config.camera.get('optimizer_kwargs', {}))

        V_torch_list = []
        F_torch_list = []
        for j in range(f_samples.shape[0]):
            f_sample = f_samples[j]
            V_j, F_j = gpytoolbox.marching_cubes(
                torch_to_numpy(f_sample), torch_to_numpy(self.x_grid), 
                self.grid_density, self.grid_density, self.grid_density, 
                0.0
            )
            V_j_torch = numpy_to_torch(V_j, self.device)
            F_j_torch = numpy_to_torch(F_j, self.device)
            V_torch_list.append(V_j_torch)
            F_torch_list.append(F_j_torch)

        R_history = []
        t_history = []
        # with torch.autograd.detect_anomaly():
        if True:
            for it in (pbar := trange(self.config.camera.num_iterations, desc="Multi-start gradient descent", leave=False, disable=not self.verbose)):
                expected_acquisitions = []
                def closure():
                    nonlocal expected_acquisitions
                    expected_acquisitions = []
                    optimizer.zero_grad()

                    eyes = eye_params / (eye_params.norm(dim=-1, keepdim=True) + 1e-8) * self.config.camera.radius
                    R_step, t_step = cameras_looking_at_origin(eyes)

                    if eyes.isnan().any() or torch.isnan(R_step).any() or torch.isnan(t_step).any():
                        raise ValueError(f"NaN values in R_step or t_step at iteration {it}")
                    if it % self.config.camera.save_every == 0:
                        R_history.append(torch_to_numpy(R_step))
                        t_history.append(torch_to_numpy(t_step))
                    for j in range(f_samples.shape[0]):
                        V_j_torch = V_torch_list[j]
                        F_j_torch = F_torch_list[j]

                        if self.multi_start_batch_size is None or self.multi_start_batch_size >= R_step.shape[0]:
                            expected_acquisition = self.scan_and_simulate(V_j_torch, F_j_torch, R_step, t_step, x_data, v_data, step_ckpt)
                        else:
                            expected_acquisition = []
                            for bi in range(0, R_step.shape[0], self.multi_start_batch_size):
                                start = bi
                                end = min(start + self.multi_start_batch_size, R_step.shape[0])
                                expected_acquisition_batch = self.scan_and_simulate(V_j_torch, F_j_torch, R_step[start:end], t_step[start:end], x_data, v_data, step_ckpt)
                                expected_acquisition.append(expected_acquisition_batch)
                            expected_acquisition = torch.cat(expected_acquisition, dim=0) # (C,)
                        assert expected_acquisition.requires_grad
                        expected_acquisitions.append(expected_acquisition)
                
                    expected_acquisitions = torch.stack(expected_acquisitions, dim=0) # (S,C)
                    expected_acquisitions = expected_acquisitions.mean(dim=0) # (C,)
                    loss = -expected_acquisitions.sum()
                    loss.backward()
                    return loss
                
                optimizer.step(closure)

                if self.verbose:
                    with torch.no_grad():
                        mean_acq = expected_acquisitions.mean().item()
                        min_acq = expected_acquisitions.min().item()
                        max_acq = expected_acquisitions.max().item()
                        postfix_str = f"Mean: {mean_acq:.4f}, Min: {min_acq:.4f}, Max: {max_acq:.4f}"
                        pbar.set_postfix_str(postfix_str)
        
        with torch.no_grad():
            R_history = np.array(R_history)
            t_history = np.array(t_history)
            step_ckpt['camera_opt_history'][self.current_step] = (R_history, t_history)
            expected_acquisitions = []
            eyes = eye_params / (eye_params.norm(dim=-1, keepdim=True) + 1e-8) * self.config.camera.radius
            R_final, t_final = cameras_looking_at_origin(eyes)
            del eye_params
            gc.collect()
            torch.cuda.empty_cache()
            for j in range(f_samples.shape[0]):
                V_j_torch = V_torch_list[j]
                F_j_torch = F_torch_list[j]
                points, normals, masks = self.simulate_scan(V_j_torch, F_j_torch, R_final, t_final)
                # next_x_data_sims, next_v_data_sims, scan_mask = self.simulate_next_data(x_data, v_data, points, normals, masks)
                next_x_data_sims, next_v_data_sims = self.simulate_next_data(x_data, v_data, points, normals, masks)
                # expected_acquisition = torch.full((masks.shape[0],), -float('inf'), device=self.device, requires_grad=True)
                expected_acquisition = self.expected_acquisition(next_x_data_sims, next_v_data_sims, step_ckpt)
                expected_acquisitions.append(expected_acquisition)
            expected_acquisitions = torch.stack(expected_acquisitions, dim=0) # (S,C)
            expected_acquisitions = expected_acquisitions.mean(dim=0) # (C,)

            selected_ci = torch.argmax(expected_acquisitions).item()
            if self.verbose:
                msg = f"Selected camera {selected_ci} for step {self.current_step+1}"
                expected_gain = expected_acquisitions[selected_ci].item()
                if self.acquisition == 'entropy':
                    msg += f", Expected next entropy: {step_ckpt['entropy_data'].item() - expected_gain}"
                else:
                    msg += f", Expected next cross entropy: {expected_gain}"
                print(msg)
            
        return R_final[selected_ci].detach(), t_final[selected_ci].detach(), step_ckpt

    def save_results(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        super().save_results(x_data, v_data, step_ckpt)
        
        # Save camera optimization history
        camera_opt_history = step_ckpt['camera_opt_history']
        camera_opt_history_data = {
            'num_steps': len(camera_opt_history),
            'K': torch_to_numpy(self.K),
        }
        for i, history in enumerate(camera_opt_history):
            Rs, ts = history
            camera_opt_history_data[f'step_{i:03d}_R'] = Rs
            camera_opt_history_data[f'step_{i:03d}_t'] = ts
        np.savez(self.result_dir / "camera_opt_history.npz", **camera_opt_history_data)