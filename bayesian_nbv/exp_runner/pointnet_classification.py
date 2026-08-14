from bayesian_nbv.exp_runner.base import *

from bayesian_nbv.pointnet.utils import model_predict, model_predict_batch, subsample_points, align_mesh
from bayesian_nbv.pointnet.pointnet2_cls_msg import PointNet2ClsMsg

class PointNetClassificationRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.point_batch_size = config.classification.point_batch_size

    def setup_exp_dir(self, args: edict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.gt_cls = os.path.basename(args.mesh_file)
        self.gt_cls = self.gt_cls[:self.gt_cls.rfind('_')]
        self.exp_name = self.acquisition + '_' + self.gt_cls + '_' + timestamp if args.exp_name is None else args.exp_name
        self.exp_dir = Path(args.exp_dir) / self.exp_name
        self.result_dir = self.exp_dir / self.exp_name
        print(f"Experiment directory: {self.exp_dir}; result directory: {self.result_dir}")
    
    def check_existing(self) -> bool:
        return (self.result_dir / "step_classes.txt").exists()

    def setup(self):
        args = self.config.classification
        with open(args.class_file, 'r') as f:
            self.classes = f.readlines()
        self.classes = [line.strip() for line in self.classes]
        self.model = PointNet2ClsMsg(num_class=len(self.classes), normal_channel=True)
        self.model.load_state_dict(torch.load(args.ckpt_path)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = torch.compile(self.model)
    
    def preprocess_mesh(self, mesh_file: str, v: np.ndarray, f: np.ndarray):
        if mesh_file.endswith('.off'):
            v, f = align_mesh(v, f)
        v = normalize_points(v) * self.config.scan.mesh_normalization_radius
        return v, f
    
    def check_init_camera(self, R: torch.Tensor, t: torch.Tensor) -> bool:
        return True
        # with torch.inference_mode():
        #     points, normals, _, mask = simulate_scan_single_camera(
        #         self.V_torch, self.F_torch, None, 
        #         R, t, self.K, 
        #         self.config.scan.resolution[0], self.config.scan.resolution[1], device=self.device, 
        #         flip_back_normals=True
        #     )
        #     points = points[mask]
        #     normals = normals[mask]
        #     choice, _ = model_predict(self.model, points, normals)
        # return self.classes[choice] != self.gt_cls
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['step_classes'] = []
        step_ckpt['log_scores_data_prev'] = None
        return step_ckpt
    
    @torch.inference_mode()
    def step_utility(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        if (subsample := self.config.classification.get('subsample_points', None)) is not None:
            x_data, v_data = subsample_points(x_data, v_data, subsample, random_start_point=True)
        else:
            if x_data.shape[0] < 128:
                x_data, v_data = subsample_points(x_data, v_data, 128)
            elif self.max_points is not None and x_data.shape[0] > self.max_points:
                x_data, v_data = subsample_points(x_data, v_data, self.max_points, random_start_point=True)

        choice_data, log_scores_data = model_predict(self.model, x_data, v_data)
        entropy_data = -torch.sum(log_scores_data * torch.exp(log_scores_data))
        
        msg = f"Step {self.current_step:03d}, GT class: {self.gt_cls}, Predicted class: {self.classes[choice_data]}, Entropy: {entropy_data.item()}"
        step_ckpt['step_classes'].append(self.classes[choice_data])
        step_ckpt['entropy_data'] = entropy_data.detach()
        if step_ckpt['log_scores_data_prev'] is not None:
            cross_entropy_data = -torch.sum(log_scores_data * torch.exp(step_ckpt['log_scores_data_prev']))
            msg += f", Cross entropy with previous step: {cross_entropy_data.item()}"
        if self.verbose:
            print(msg)
        step_ckpt['log_scores_data_prev'] = log_scores_data.detach()
        return step_ckpt
    
    def expected_acquisition(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        """
        x_data: (C,N,3)
        v_data: (C,N,3)
        step_ckpt: dict[str, Any]
        """
        grad_ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_ctx:
            # x_data_np = torch_to_numpy(x_data)
            # v_data_np = torch_to_numpy(v_data)
            if self.point_batch_size is None:
                _, log_scores_samples = model_predict(self.model, x_data, v_data)
            else:
                n_batches = (x_data.shape[0] + self.point_batch_size - 1) // self.point_batch_size
                log_scores_samples = []
                for i in range(n_batches):
                    start = i * self.point_batch_size
                    end = min(start + self.point_batch_size, x_data.shape[0])
                    x_batch = x_data[start:end]
                    v_batch = v_data[start:end]
                    _, log_scores_batch = model_predict_batch(self.model, x_batch, v_batch, detach=not requires_grad)
                    log_scores_samples.append(log_scores_batch)
                log_scores_samples = torch.cat(log_scores_samples, dim=0) # (C, N_class)
            
            # log_score_samples_np = torch_to_numpy(log_scores_samples)
            # np.savez_compressed(f'debug/debug_{self.current_step:03d}.npz', x_data=x_data_np, v_data=v_data_np, log_scores_samples=log_score_samples_np)
            
            if self.acquisition == 'entropy':
                entropy_samples = -torch.sum(log_scores_samples * torch.exp(log_scores_samples), dim=1)
                acquisition = torch.maximum(step_ckpt['entropy_data'] - entropy_samples, torch.zeros_like(entropy_samples))
            elif self.acquisition == 'cross_entropy':
                cross_entropy_samples = -torch.sum(log_scores_samples * torch.exp(step_ckpt['log_scores_data_prev'].unsqueeze(0)), dim=1)
                acquisition = cross_entropy_samples
            elif self.acquisition == 'mixed':
                entropy_threshold = self.config.classification.entropy_threshold
                if step_ckpt['entropy_data'] > entropy_threshold:
                    acquisition = -torch.sum(log_scores_samples * torch.exp(step_ckpt['log_scores_data_prev'].unsqueeze(0)), dim=1)
                else:
                    entropy_samples = -torch.sum(log_scores_samples * torch.exp(log_scores_samples), dim=1)
                    acquisition = torch.maximum(step_ckpt['entropy_data'] - entropy_samples, torch.zeros_like(entropy_samples))
            else:
                raise ValueError(f"Invalid acquisition function: {self.acquisition}")
        
        return acquisition # (C,)
    
    def save_results(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        super().save_results(x_data, v_data, step_ckpt)
        
        # Save step class labels
        step_classes = step_ckpt['step_classes']
        with open(self.result_dir / "step_classes.txt", "w") as f:
            for cls in step_classes:
                f.write(f"{cls}\n")
        
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
    
    
    def verbose_msg(self, msg, step_ckpt, expected_acqisition):
        if self.acquisition == 'entropy':
            msg += f", Expected next entropy: {step_ckpt['entropy_data'].item() - expected_acqisition}"
        # elif self.acquisition == 'cross_entropy':
        elif self.acquisition == 'cross_entropy':
            msg += f", Expected next cross entropy: {expected_acqisition}"
        return msg