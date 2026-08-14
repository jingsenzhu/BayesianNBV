from bayesian_nbv.exp_runner.base import *
import robust_laplacian
from bayesian_nbv.laplacian.implicit_euler import solve_heat_with_source
import point_cloud_utils as pcu

class HeatDiffusionRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)

    def setup_exp_dir(self, args: edict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = Path(args.mesh_file).stem + '_' + timestamp if args.exp_name is None else args.exp_name
        self.exp_dir = Path(args.exp_dir) / self.exp_name
        self.result_dir = self.exp_dir / self.exp_name
        print(f"Experiment directory: {self.exp_dir}; result directory: {self.result_dir}")
    
    def check_existing(self) -> bool:
        return (self.result_dir / "step_simulation.npz").exists()
    
    def setup(self):
        args = self.config.heat_diffusion
        self.heat_source = args.heat_source
        self.T = args.T
        self.n_euler_steps = args.n_euler_steps
        self.u_init_cfg = args.init_temperature
    
    def preprocess_mesh(self, mesh_file: str, v: np.ndarray, f: np.ndarray):
        v = normalize_points(v) * self.config.scan.mesh_normalization_radius
        if self.heat_source.type == 'random_point':
            heat_source_idx = np.random.randint(0, v.shape[0])
            self.heat_source_center = v[heat_source_idx:heat_source_idx+1]
            heat_source_mask = (np.linalg.norm(v - self.heat_source_center, axis=1) <= self.heat_source.radius)
            self.f_vert = np.zeros_like(v[:, 0])
            self.f_vert[heat_source_mask] = self.heat_source.power
            self.heat_source_points = v[heat_source_mask]
            normals = pcu.estimate_mesh_vertex_normals(v, f)
            self.heat_source_normals = normals[heat_source_mask]
            self.heat_source_idx = heat_source_idx
        elif self.heat_source.type == 'fixed_point':
            heat_source_idx = self.heat_source.vert_idx
            self.heat_source_center = v[heat_source_idx:heat_source_idx+1]
            heat_source_mask = (np.linalg.norm(v - self.heat_source_center, axis=1) <= self.heat_source.radius)
            self.f_vert = np.zeros_like(v[:, 0])
            self.f_vert[heat_source_mask] = self.heat_source.power
            self.heat_source_points = v[heat_source_mask]
            normals = pcu.estimate_mesh_vertex_normals(v, f)
            self.heat_source_normals = normals[heat_source_mask]
            self.heat_source_idx = heat_source_idx
        else:
            raise NotImplementedError(f"Invalid heat source type: {self.heat_source.type}")
        
        print(f"Heat source vertex index: {self.heat_source_idx}")
        
        self.L_ref, self.M_ref = robust_laplacian.mesh_laplacian(v, f)

        if self.u_init_cfg.type == 'constant':
            # self.u_init = self.u_init_cfg.value * np.ones_like(v[:, 0])
            self.u_init_v = np.full_like(v[:, 0], self.u_init_cfg.value)
        else:
            raise NotImplementedError(f"Invalid initial temperature type: {self.u_init_cfg.type}")
        
        self.u_ref, _ = solve_heat_with_source(self.L_ref, self.M_ref, self.u_init_v, self.f_vert, self.T, self.n_euler_steps)
        self.ref_coldest = v[self.u_ref.argmin()]
        
        return v, f
    
    def load_mesh_and_init_scan(self, args: edict, mesh_file: str):
        points, normals, R_init, t_init = super().load_mesh_and_init_scan(args, mesh_file)
        points = torch.cat([points, numpy_to_torch(self.heat_source_points, self.device)], dim=0)
        normals = torch.cat([normals, numpy_to_torch(self.heat_source_normals, self.device)], dim=0)
        return points, normals, R_init, t_init
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['step_simulation'] = []
        step_ckpt['step_f'] = []
        step_ckpt['step_coldest'] = []
        step_ckpt['step_x_data'] = []
        return step_ckpt

    def get_heat_source_and_init(self, x):
        """
        x: (N, 3)
        """
        if self.heat_source.type == 'random_point' or self.heat_source.type == 'fixed_point':
            heat_source_mask = (np.linalg.norm(x - self.heat_source_center, axis=1) <= self.heat_source.radius)
            f_x = np.zeros_like(x[:, 0])
            f_x[heat_source_mask] = self.heat_source.power
        else:
            raise NotImplementedError(f"Invalid heat source type: {self.heat_source.type}")
        if self.u_init_cfg.type == 'constant':
            u_init_x = np.full_like(x[:, 0], self.u_init_cfg.value)
        else:
            raise NotImplementedError(f"Invalid initial temperature type: {self.u_init_cfg.type}")
        return f_x, u_init_x
    
    def step_utility(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        if self.max_points is not None and x_data.shape[0] > self.max_points:
            x_data, v_data = subsample_points(x_data, v_data, self.max_points)
        
        x_data_np = torch_to_numpy(x_data) # (N, 3)

        f_x, u_init_x = self.get_heat_source_and_init(x_data_np)
        
        L_x, M_x = robust_laplacian.point_cloud_laplacian(x_data_np)

        u_x, _ = solve_heat_with_source(L_x, M_x, u_init_x, f_x, self.T, self.n_euler_steps)
        step_ckpt['step_x_data'].append(x_data_np)
        step_ckpt['step_simulation'].append(u_x)
        step_ckpt['step_f'].append(f_x)
        step_ckpt['neg_min_u'] = -u_x.min()
        x_coldest = x_data_np[u_x.argmin()]
        step_ckpt['step_coldest'].append(x_coldest)

        if self.verbose:
            dist_to_ref_coldest = np.linalg.norm(x_coldest - self.ref_coldest)
            msg = f"Step {self.current_step:03d}, Reference minimum temperature: {self.u_ref.min()}, Minimum temperature from current scan: {u_x.min()}, Simulated coldest point's distance to reference: {dist_to_ref_coldest}"
            print(msg)

        return step_ckpt
    
    def expected_acquisition(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any], requires_grad: bool = False):
        """
        x_data: (N,3)
        v_data: (N,3)
        step_ckpt: dict[str, Any]
        """
        x_hat = torch_to_numpy(x_data) # (N, 3)

        f_x_hat, u_init_x_hat = self.get_heat_source_and_init(x_hat)
        L_x_hat, M_x_hat = robust_laplacian.point_cloud_laplacian(x_hat)
        u_x_hat, _ = solve_heat_with_source(L_x_hat, M_x_hat, u_init_x_hat, f_x_hat, self.T, self.n_euler_steps)
        neg_min_u_hat = -u_x_hat.min()
        return max(neg_min_u_hat, step_ckpt['neg_min_u']) # scalar
    
    def save_results(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        super().save_results(x_data, v_data, step_ckpt)

        # Save step scans and simulation results
        step_scans = step_ckpt['step_scans']
        step_simulation = step_ckpt['step_simulation']
        step_f = step_ckpt['step_f']
        step_x_data = step_ckpt['step_x_data']
        scan_data = {
            'num_steps': len(step_scans),
            'heat_source_idx': self.heat_source_idx,
            'f_ref': self.f_vert,
            'u_ref': self.u_ref
        }
        for i in range(len(step_scans)):
            scan_data[f'step_{i:03d}'] = step_scans[i]
            scan_data[f'step_{i:03d}_u'] = step_simulation[i]
            scan_data[f'step_{i:03d}_x'] = step_x_data[i]
            scan_data[f'step_{i:03d}_f'] = step_f[i]
        step_coldest = step_ckpt['step_coldest']
        step_coldest = np.array(step_coldest)
        scan_data['step_coldest'] = step_coldest
        np.savez(self.result_dir / "scan_data.npz", **scan_data)

        # Save selected cameras
        Rs_selected = np.array(step_ckpt['Rs_selected'])
        ts_selected = np.array(step_ckpt['ts_selected'])
        np.savez(self.result_dir / "selected_cameras.npz", Rs=Rs_selected, ts=ts_selected)

        # Save final scan
        x_data_np = torch_to_numpy(x_data)
        v_data_np = torch_to_numpy(v_data)
        pcu.save_mesh_vn(str(self.result_dir / "final_scan.ply"), x_data_np, v_data_np)

        # Compare with reference simulation
        u_min_ref = self.u_ref.min()
        u_min_final = step_simulation[-1].min()
        x_coldest_final = step_coldest[-1]
        dist_to_ref_coldest_final = np.linalg.norm(x_coldest_final - self.ref_coldest)
        print(f"Reference minimum temperature: {u_min_ref}, Minimum temperature from final scan: {u_min_final}, Simulated coldest point's distance to reference: {dist_to_ref_coldest_final}")