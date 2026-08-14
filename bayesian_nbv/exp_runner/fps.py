from bayesian_nbv.exp_runner.base import *
from bayesian_nbv.scan.camera import cameras_on_sphere_lookat_origin, farthest_point_sampling
from bayesian_nbv.pointnet.utils import subsample_points
from numpy.random import default_rng

class FPSRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.needs_sample = False
        self.start_indices = []
        self.cam_idx_rng = default_rng(config.camera.get('camera_idx_seed', 0))
    
    def prepare_init_camera(self, args: edict) -> tuple[torch.Tensor, torch.Tensor]:
        self.R_all, self.t_all, eye_all = cameras_on_sphere_lookat_origin(args.num_cameras, args.radius, device=self.device)

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
        self.selected_indices = farthest_point_sampling(eye_all, self.num_steps, start_idx=start_idx)

        np.savez(self.result_dir / "all_cameras.npz", R=torch_to_numpy(self.R_all), t=torch_to_numpy(self.t_all), K=torch_to_numpy(self.K))
        return self.R_all[start_idx], self.t_all[start_idx]
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['selected_indices'] = self.selected_indices.tolist()
        return step_ckpt
    
    def expected_improvement(self, f_samples: torch.Tensor, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        selected_idx = self.selected_indices[self.current_step + 1]
        return self.R_all[selected_idx], self.t_all[selected_idx], step_ckpt