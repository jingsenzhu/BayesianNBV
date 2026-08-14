from bayesian_nbv.exp_runner.base import *
from pytorch3d.loss import chamfer_distance
from bayesian_nbv.pointnet.utils import align_mesh_shapenet, align_mesh
import trimesh

class ChamferRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.eval_n_points = config.chamfer.eval_n_points
        self.subsample_samples = False

    def setup_exp_dir(self, args: edict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = 'chamfer_' + os.path.basename(args.mesh_file) + timestamp if args.exp_name is None else args.exp_name
        self.exp_dir = Path(args.exp_dir) / self.exp_name
        self.result_dir = self.exp_dir / self.exp_name
        print(f"Experiment directory: {self.exp_dir}; result directory: {self.result_dir}")
    
    def check_existing(self) -> bool:
        return (self.result_dir / "final_scan.ply").exists()
    
    def setup(self):
        pass
    
    def preprocess_mesh(self, mesh_file: str, v: np.ndarray, f: np.ndarray):
        # if not trimesh.Trimesh(v, f).is_watertight:
        #     v, f = pcu.make_mesh_watertight(v, f, 20000)
        if mesh_file.endswith('.off'):
            v, f = align_mesh(v, f)
        elif mesh_file.endswith('.obj'):
            v, f = align_mesh_shapenet(v, f)
        v = normalize_points(v) * self.config.scan.mesh_normalization_radius
        return v, f
    
    def check_init_camera(self, R: torch.Tensor, t: torch.Tensor) -> bool:
        return True
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['step_reconstruction_losses'] = []
        return step_ckpt
    
    def step_utility(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        return step_ckpt
    
    def post_reconstruction(self, V_step: np.ndarray, F_step: np.ndarray, step_ckpt: dict[str, Any]):
        V_gt = torch_to_numpy(self.V_torch)
        F_gt = torch_to_numpy(self.F_torch)

        fid, bc = pcu.sample_mesh_poisson_disk(V_gt, F_gt, num_samples=self.eval_n_points)
        points_gt = pcu.interpolate_barycentric_coords(F_gt, fid, bc, V_gt)

        fid, bc = pcu.sample_mesh_poisson_disk(V_step, F_step, num_samples=self.eval_n_points)
        points_step = pcu.interpolate_barycentric_coords(F_step, fid, bc, V_step)
        points_step = points_step.astype(points_gt.dtype)

        cd = pcu.chamfer_distance(points_gt, points_step)
        hd = pcu.hausdorff_distance(points_gt, points_step)

        # fid, bc = pcu.sample_mesh_poisson_disk(V_gt, F_gt, num_samples=self.eval_n_points // 2)
        # points_gt_half = pcu.interpolate_barycentric_coords(F_gt, fid, bc, V_gt)
        # points_gt_half = points_gt_half.astype(np.float64)
        # fid, bc = pcu.sample_mesh_poisson_disk(V_step, F_step, num_samples=self.eval_n_points // 2)
        # points_step_half = pcu.interpolate_barycentric_coords(F_step, fid, bc, V_step)
        # points_step_half = points_step_half.astype(np.float64)

        # emd, _ = pcu.earth_movers_distance(points_gt_half, points_step_half)

        if self.verbose:
            print(f"Reconstruction for step {self.current_step:03d}, Chamfer distance: {cd:.6f}, Hausdorff distance: {hd:.6f}")

        step_ckpt['step_reconstruction_losses'].append((cd, hd))

        return step_ckpt
    
    def expected_acquisition(self, x_data: list[torch.Tensor], v_data: list[torch.Tensor], step_ckpt: dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        """
        x_data: list of (N_i,3)
        v_data: list of (N_i,3)
        step_ckpt: dict[str, Any]
        """
        N_curr_points = self.x_data_current.shape[0]
        max_points_num = np.max([x.shape[0] - N_curr_points for x in x_data])
        pts_lengths = []
        scanned_points = torch.zeros(len(x_data), max_points_num, 3, device=self.device)
        for i in range(len(x_data)):
            N_scan = x_data[i].shape[0] - N_curr_points
            pts_lengths.append(N_scan)
            scanned_points[i,:N_scan] = x_data[i][:-N_curr_points]
        pts_lengths = torch.tensor(pts_lengths, dtype=torch.long, device=self.device)
        curr_points = self.x_data_current.unsqueeze(0).expand(len(x_data), -1, 3)
        acquisition, _ = chamfer_distance(
            scanned_points, curr_points, x_lengths=pts_lengths, batch_reduction=None, single_directional=True
        )
        return acquisition

    def save_results(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        super().save_results(x_data, v_data, step_ckpt)
        step_reconstruction_losses = step_ckpt['step_reconstruction_losses']
        with open(self.result_dir / "step_reconstruction_losses.txt", "w") as f:
            for cd, hd in step_reconstruction_losses:
                f.write(f"{cd:.6f} {hd:.6f}\n")
        
        # Save step scans
        step_scans = step_ckpt['step_scans']
        scan_data = {
            'num_steps': len(step_scans)
        }
        for i, scan in enumerate(step_scans):
            scan_data[f'step_{i:03d}'] = scan
        np.savez(self.result_dir / "scan_data.npz", **scan_data)

        # Save selected cameras
        Rs_selected = np.array(step_ckpt['Rs_selected'])
        ts_selected = np.array(step_ckpt['ts_selected'])
        np.savez(self.result_dir / "selected_cameras.npz", Rs=Rs_selected, ts=ts_selected)

        # Save final scan
        x_data_np = torch_to_numpy(x_data)
        v_data_np = torch_to_numpy(v_data)
        pcu.save_mesh_vn(str(self.result_dir / "final_scan.ply"), x_data_np, v_data_np)