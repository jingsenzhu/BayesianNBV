import torch
import torch.nn as nn
import torch.optim as optim
import gc
from bayesian_nbv.exp_runner.base import *
from bayesian_nbv.scan.camera import cameras_on_sphere_lookat_origin
from bayesian_nbv.pointnet.utils import subsample_points
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

class IterativeMultiStartGradientDescentRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.subsample_points = self.config.points.subsample_points
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
        R_starts, t_starts, _ = cameras_on_sphere_lookat_origin(self.config.camera.num_starts, self.config.camera.radius, device=self.device, random_oriented=True)
        # R_params = nn.Parameter(matrix_to_rotation_6d(R_starts))
        # t_params = nn.Parameter(t_starts)
        # optimizer = optim.Adam([R_params, t_params], lr=self.config.camera.learning_rate)

        # optimizer = getattr(optim, self.config.camera.optimizer)([R_params, t_params], lr=self.config.camera.learning_rate, **self.config.camera.get('optimizer_kwargs', {}))

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

        R_history_batches = []
        t_history_batches = []
        R_final = []
        t_final = []

        assert self.multi_start_batch_size is not None
        for bi in range(0, R_starts.shape[0], self.multi_start_batch_size):
            start = bi
            end = min(start + self.multi_start_batch_size, R_starts.shape[0])
            t_starts_batch = t_starts[start:end]
            R_starts_batch = R_starts[start:end]
            R_params = nn.Parameter(matrix_to_rotation_6d(R_starts_batch))
            t_params = nn.Parameter(t_starts_batch)
            optimizer = getattr(optim, self.config.camera.optimizer)([R_params, t_params], lr=self.config.camera.learning_rate, **self.config.camera.get('optimizer_kwargs', {}))

            R_history_batch = []
            t_history_batch = []
            expected_acquisitions_final = []

            # TODO: implement true iterative multi-start gradient descent, separate backward pass for each camera batch
            for it in (pbar := trange(self.config.camera.num_iterations, desc="Multi-start gradient descent", leave=False, disable=not self.verbose)):
                expected_acquisitions_batch = []
                def closure():
                    nonlocal expected_acquisitions_batch
                    expected_acquisitions_batch = []
                    optimizer.zero_grad()
                    R_step = rotation_6d_to_matrix(R_params)
                    t_step = t_params
                    if it % self.config.camera.save_every == 0:
                        # step_ckpt['camera_opt_history'][self.current_step].append((R_step.detach().cpu().numpy(), t_step.detach().cpu().numpy()))
                        R_history_batch.append(torch_to_numpy(R_step))
                        t_history_batch.append(torch_to_numpy(t_step))
                    for j in range(f_samples.shape[0]):
                        V_j_torch = V_torch_list[j]
                        F_j_torch = F_torch_list[j]

                        expected_acquisition = self.scan_and_simulate(V_j_torch, F_j_torch, R_step, t_step, x_data, v_data, step_ckpt)
                        assert expected_acquisition.requires_grad
                        expected_acquisitions_batch.append(expected_acquisition)
                
                    expected_acquisitions_batch = torch.stack(expected_acquisitions_batch, dim=0) # (S,C)
                    expected_acquisitions_batch = expected_acquisitions_batch.mean(dim=0) # (C,)
                    loss = -expected_acquisitions_batch.sum()
                    loss.backward()
                    return loss
                
                optimizer.step(closure)

                if self.verbose:
                    with torch.no_grad():
                        mean_acq = expected_acquisitions_batch.mean().item()
                        min_acq = expected_acquisitions_batch.min().item()
                        max_acq = expected_acquisitions_batch.max().item()
                        postfix_str = f"Mean: {mean_acq:.4f}, Min: {min_acq:.4f}, Max: {max_acq:.4f}"
                        pbar.set_postfix_str(postfix_str)
            
            with torch.no_grad():
                R_history_batch = np.array(R_history_batch) # (I,B,3,3)
                t_history_batch = np.array(t_history_batch) # (I,B,3)
                R_history_batches.append(R_history_batch)
                t_history_batches.append(t_history_batch)
                R_final_batch = rotation_6d_to_matrix(R_params).detach()
                t_final_batch = t_params.detach()
                R_final.append(R_final_batch)
                t_final.append(t_final_batch)
                del R_params, t_params
                gc.collect()
                torch.cuda.empty_cache()
                expected_acquisitions_batch_final = []
                for j in range(f_samples.shape[0]):
                    V_j_torch = V_torch_list[j]
                    F_j_torch = F_torch_list[j]
                    expected_acquisition_batch_final = self.scan_and_simulate(V_j_torch, F_j_torch, R_final_batch, t_final_batch, x_data, v_data, step_ckpt)
                    expected_acquisitions_batch_final.append(expected_acquisition_batch_final)
                expected_acquisitions_batch_final = torch.stack(expected_acquisitions_batch_final, dim=0) # (S,B)
                expected_acquisitions_batch_final = expected_acquisitions_batch_final.mean(dim=0) # (B,)
                expected_acquisitions_final.append(expected_acquisitions_batch_final)
        

        with torch.no_grad():
            expected_acquisitions_final = torch.cat(expected_acquisitions_final, dim=0) # (C,)
            R_final = torch.cat(R_final, dim=0) # (C,3,3)
            t_final = torch.cat(t_final, dim=0) # (C,3)
            R_history = np.concatenate(R_history_batches, axis=1) # (I,C,3,3)
            t_history = np.concatenate(t_history_batches, axis=1) # (I,C,3)

            selected_ci = torch.argmax(expected_acquisitions_final).item()
            if self.verbose:
                msg = f"Selected camera {selected_ci} for step {self.current_step+1}"
                expected_gain = expected_acquisitions_final[selected_ci].item()
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