from bayesian_nbv.exp_runner.base import *
from bayesian_nbv.scan.camera import cameras_on_sphere_lookat_origin, farthest_point_sampling
from bayesian_nbv.pointnet.utils import subsample_points

class RandomRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.needs_sample = False
    
    def prepare_init_camera(self, args: edict) -> tuple[torch.Tensor, torch.Tensor]:
        self.R_all, self.t_all, eye_all = cameras_on_sphere_lookat_origin(args.num_cameras, args.radius, device=self.device)

        population = np.delete(np.arange(args.num_cameras), args.start_idx)
        self.selected_indices = np.random.choice(population, size=self.num_steps - 1, replace=False)
        self.selected_indices = np.concatenate([[args.start_idx], self.selected_indices])

        np.savez(self.result_dir / "all_cameras.npz", R=torch_to_numpy(self.R_all), t=torch_to_numpy(self.t_all), K=torch_to_numpy(self.K))
        return self.R_all[args.start_idx], self.t_all[args.start_idx]
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['selected_indices'] = self.selected_indices.tolist()
        return step_ckpt
    
    def expected_improvement(self, f_samples: torch.Tensor, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        selected_idx = self.selected_indices[self.current_step]
        return self.R_all[selected_idx], self.t_all[selected_idx], step_ckpt