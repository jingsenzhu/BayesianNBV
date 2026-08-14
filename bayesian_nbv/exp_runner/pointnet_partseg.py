from bayesian_nbv.exp_runner.base import *

from bayesian_nbv.pointnet.utils import model_predict_partseg, model_predict_partseg_batch, subsample_points, align_mesh_shapenet, NUM_CLASSES, NUM_PARTS, SEG_CLASSES
from bayesian_nbv.pointnet.pointnet2_partseg_msg import PointNet2PartSegMsg

# SEG_CLASSES = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

# SEG_LABEL_TO_CAT = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in SEG_CLASSES.keys():
#     for label in SEG_CLASSES[cat]:
#         SEG_LABEL_TO_CAT[label] = cat


# CAT_ID = {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
CAT_ID = {'02691156': 'Airplane', '02773838': 'Bag', '02954340': 'Cap', '02958343': 'Car', '03001627': 'Chair', '03261776': 'Earphone', '03467517': 'Guitar', '03624134': 'Knife', '03636649': 'Lamp', '03642806': 'Laptop', '03790512': 'Motorbike', '03797390': 'Mug', '03948459': 'Pistol', '04099429': 'Rocket', '04225987': 'Skateboard', '04379243': 'Table'}


class PointNetPartSegmentationRunner(BaseRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
        self.point_batch_size = config.partseg.point_batch_size
        self.target_points = config.partseg.target_points

    def setup_exp_dir(self, args: edict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mesh_path = Path(args.mesh_file) # .../<cat_id>/<shape_id>/models/model_normalized.obj
        cat_id = mesh_path.parent.parent.parent.name
        shape_id = mesh_path.parent.parent.name
        self.shape_id = shape_id
        self.cat_id = cat_id
        self.cls_label = CAT_ID[cat_id]
        self.exp_name = self.cls_label + '_' + shape_id + '_' + timestamp if args.exp_name is None else args.exp_name
        self.exp_dir = Path(args.exp_dir) / self.exp_name
        self.result_dir = self.exp_dir / self.exp_name
        print(f"Experiment directory: {self.exp_dir}; result directory: {self.result_dir}")
    
    def check_existing(self) -> bool:
        return (self.result_dir / "step_parts.txt").exists()

    def setup(self):
        args = self.config.partseg
        self.model = PointNet2PartSegMsg(num_classes=NUM_PARTS, normal_channel=True)
        self.model.load_state_dict(torch.load(args.ckpt_path)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = torch.compile(self.model)
    
    def preprocess_mesh(self, mesh_file: str, v: np.ndarray, f: np.ndarray):
        v, f = align_mesh_shapenet(v, f)
        v = normalize_points(v) * self.config.scan.mesh_normalization_radius
        return v, f
    
    def check_init_camera(self, R: torch.Tensor, t: torch.Tensor) -> bool:
        return True
    
    def init_step_checkpoint(self, **kwargs) -> dict[str, Any]:
        step_ckpt = super().init_step_checkpoint(**kwargs)
        step_ckpt['step_segments'] = []
        step_ckpt['step_parts'] = []
        if (data_path := self.config.partseg.get('data_path', None)) is not None:
            data_path = Path(data_path) / self.cat_id / (self.shape_id + '.txt')
            gt_data = np.loadtxt(data_path)
            x_gt = gt_data[:, :3]
            x_gt = normalize_points(x_gt) * self.config.scan.mesh_normalization_radius
            v_gt = gt_data[:, 3:6]
            part_labels = gt_data[:, 6].astype(np.int32)
            seg_classes = SEG_CLASSES[self.cls_label]
            gt_segments = []
            for part in seg_classes:
                part_mask = part_labels == part
                x_part = x_gt[part_mask]
                v_part = v_gt[part_mask]
                gt_segments.append((x_part, v_part))
            step_ckpt['gt_segments'] = gt_segments

        return step_ckpt
    
    @torch.inference_mode()
    def step_utility(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        x_data, v_data = subsample_points(x_data, v_data, self.subsample_points)
        softmax_scores_data = model_predict_partseg(self.model, x_data, v_data, self.cls_label) # (N_points, N_parts)
        n_pts, n_parts = softmax_scores_data.shape

        soft_part_cnt = softmax_scores_data.sum(dim=0) # (N_parts,)
        utility = torch.sum(torch.tanh(soft_part_cnt / self.target_points))
        step_ckpt['utility'] = utility.detach()

        choices = softmax_scores_data.argmax(dim=1) # (N_points,)
        part_cnt = torch.bincount(choices, minlength=n_parts).long() # (N_parts,)
        valid_parts = torch.sum(part_cnt > self.target_points)
        step_ckpt['step_parts'].append(part_cnt.cpu().numpy())

        msg = f"Step {self.current_step:03d}, {valid_parts.item()} parts are found with more than {self.target_points} points, utility function value: {utility.item()}"
        if self.verbose:
            print(msg)

        x_parts, v_parts = [], []
        for i in range(n_parts):
            # if part_cnt[i] == 0:
            #     continue
            x_part = x_data[choices == i].cpu().numpy()
            v_part = v_data[choices == i].cpu().numpy()
            x_parts.append(x_part)
            v_parts.append(v_part)
        step_ckpt['step_segments'].append((x_parts, v_parts))

        return step_ckpt
    
    def expected_acquisition(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        """
        x_data: (C,N,3)
        v_data: (C,N,3)
        step_ckpt: dict[str, Any]
        """
        grad_ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_ctx:
            if self.point_batch_size is None:
                softmax_scores_samples = model_predict_partseg_batch(self.model, x_data, v_data, self.cls_label, detach=not requires_grad) # (C, N_pts, N_parts)
            else:
                n_batches = (x_data.shape[0] + self.point_batch_size - 1) // self.point_batch_size
                softmax_scores_samples = []
                for i in range(n_batches):
                    start = i * self.point_batch_size
                    end = min(start + self.point_batch_size, x_data.shape[0])
                    x_batch = x_data[start:end]
                    v_batch = v_data[start:end]
                    softmax_scores_batch = model_predict_partseg_batch(self.model, x_batch, v_batch, self.cls_label, detach=not requires_grad)
                    softmax_scores_samples.append(softmax_scores_batch)
                softmax_scores_samples = torch.cat(softmax_scores_samples, dim=0) # (C, N_pts, N_parts)
            
            soft_part_cnt_samples = softmax_scores_samples.sum(dim=1) # (C, N_parts)
            utility_samples = torch.sum(torch.tanh(soft_part_cnt_samples / self.target_points), dim=1) # (C,)

            acquisition = torch.maximum(utility_samples - step_ckpt['utility'], torch.zeros_like(utility_samples))
        
        return acquisition # (C,)
    
    def save_results(self, x_data: torch.Tensor, v_data: torch.Tensor, step_ckpt: dict[str, Any]):
        super().save_results(x_data, v_data, step_ckpt)
        
        # Save step part counts
        step_parts = step_ckpt['step_parts']
        step_parts = np.array(step_parts, dtype=np.int32) # (N_steps, N_parts)
        np.savetxt(self.result_dir / "step_parts.txt", step_parts, fmt='%d')

        # Save step segments
        step_segments = step_ckpt['step_segments']
        segment_data = {
            'num_steps': len(step_segments),
            'num_parts': len(SEG_CLASSES[self.cls_label])
        }
        for i, (x_parts, v_parts) in enumerate(step_segments):
            n_parts = len(x_parts)
            segment_data[f'num_parts_{i:03d}'] = n_parts
            for j, (x_part, v_part) in enumerate(zip(x_parts, v_parts)):
                if len(x_part) == 0:
                    segment_data[f'step_{i:03d}_part_{j:03d}'] = np.zeros((0, 6))
                    continue
                part = np.concatenate([np.asarray(x_part), np.asarray(v_part)], axis=-1) # (N_points, 6)
                segment_data[f'step_{i:03d}_part_{j:03d}'] = part
        np.savez(self.result_dir / "step_segments.npz", **segment_data)

        if 'gt_segments' in step_ckpt:
            gt_segments = step_ckpt['gt_segments']
            segment_data = {
                'num_parts': len(gt_segments)
            }
            for i, (x_part, v_part) in enumerate(gt_segments):
                part = np.concatenate([np.asarray(x_part), np.asarray(v_part)], axis=-1) # (N_points, 6)
                segment_data[f'gt_part_{i:03d}'] = part
            np.savez(self.result_dir / "gt_segments.npz", **segment_data)
        
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
        msg += f", Expected next utility: {expected_acqisition + step_ckpt['utility'].item()}"
        return msg