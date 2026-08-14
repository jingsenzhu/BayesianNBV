import argparse
import yaml
from easydict import EasyDict as edict
from pathlib import Path
from tqdm import tqdm
import shutil
import traceback

from bayesian_nbv.exp_runner.sphere_search import SphereSearchRunner
from bayesian_nbv.exp_runner.multi_start_gd import MultiStartGradientDescentRunner
from bayesian_nbv.exp_runner.sphere_msgd import SphereGradientDescentRunner
from bayesian_nbv.exp_runner.fps import FPSRunner
from bayesian_nbv.exp_runner.random import RandomRunner
from bayesian_nbv.exp_runner.uncertainty import RayUncertaintyRunner
from bayesian_nbv.exp_runner.chamfer import ChamferRunner

class ChamferSphereSearchRunner(SphereSearchRunner, ChamferRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)

class ChamferFPSRunner(FPSRunner, ChamferRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
    
    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)

class ChamferRandomRunner(RandomRunner, ChamferRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
    
    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)

class ChamferRayUncertaintyRunner(RayUncertaintyRunner, ChamferRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
    
    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)

class ChamferMSGDRunner(MultiStartGradientDescentRunner, ChamferRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
    
    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)

class ChamferSphereGradientDescentRunner(SphereGradientDescentRunner, ChamferRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)
    
    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Path to the experiment directory')
    parser.add_argument('config', type=str, help='Path to the config file')
    parser.add_argument('-mf', '--mesh_file', type=str, default=None, help='Path to the mesh file')
    parser.add_argument('-ml', '--mesh_list', type=str, default=None, help='Path to the mesh list file')
    parser.add_argument('-dt', '--dataset_type', type=str, choices=['shapenet', 'modelnet'], default='shapenet', help='Dataset type')
    parser.add_argument('-d', '--mesh_dir', type=str, default=None, help='Path to the mesh directory')
    parser.add_argument('-s', '--split', type=str, choices=['train', 'test'], default='train', help='Dataset split to use')
    parser.add_argument('-sil', '--start_idx_list', type=str, default=None, help='Start indices list file for selection')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index for camera selection')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--exp_type', type=str, choices=['sphere', 'fps', 'random', 'uncertainty', 'msgd', 'sphere_msgd'], default='sphere', help='Experiment type')
    parser.add_argument('--start_end', type=int, nargs=2, default=None, help='Start and end indices for experiment')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)

    if args.start_idx is not None:
        config.camera.start_idx = args.start_idx

    assert args.mesh_file is not None or args.mesh_list is not None, "Either mesh_file or mesh_list must be provided"
    if args.exp_type == 'sphere':
        runner = ChamferSphereSearchRunner(config)
    elif args.exp_type == 'fps':
        runner = ChamferFPSRunner(config)
    elif args.exp_type == 'random':
        runner = ChamferRandomRunner(config)
    elif args.exp_type == 'uncertainty':
        runner = ChamferRayUncertaintyRunner(config)
    elif args.exp_type == 'msgd':
        runner = ChamferMSGDRunner(config)
    elif args.exp_type == 'sphere_msgd':
        runner = ChamferSphereGradientDescentRunner(config)
    else:
        raise ValueError(f"Invalid experiment type: {args.exp_type}")

    if args.mesh_file is not None:
        exp_args = edict(
            exp_dir=args.exp_dir,
            exp_name=args.exp_name,
            seed=args.seed,
            mesh_file=args.mesh_file,
        )
        runner.run(exp_args)
    else:
        runner.verbose = False
        root_dir = Path(args.exp_dir) / 'results'
        root_dir.mkdir(parents=True, exist_ok=True)
        with open(args.mesh_list, 'r') as f:
            mesh_list = f.readlines()
        if args.dataset_type == 'shapenet':
            datapath = [Path(line.strip()) for line in mesh_list]
            shape_ids = [path.parent.parent.name for path in datapath]
        else:
            shape_ids = [line.strip() for line in mesh_list]
            shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
            datapath = [str(Path(args.mesh_dir) / shape_names[i] / args.split / (shape_ids[i] + '.off')) for i in range(len(shape_ids))]


        if args.start_idx_list is not None:
            with open(args.start_idx_list, 'r') as f:
                start_idx_list = f.readlines()
            start_idx_list = [int(line.strip()) for line in start_idx_list]
            assert len(start_idx_list) == len(datapath), "Number of start indices must be equal to number of test meshes"
        else:
            start_idx_list = [None] * len(datapath)
        
        if args.start_end is not None:
            print(f"[INFO] Starting from {args.start_end[0]} to {args.start_end[1]}")
            start_idx_list = start_idx_list[args.start_end[0]:args.start_end[1]]
            datapath = datapath[args.start_end[0]:args.start_end[1]]
        
        pbar = tqdm(zip(shape_ids, datapath, start_idx_list), desc="Processing meshes", total=len(datapath))
        for shape_id, path, start_idx in pbar:
            postfix = shape_id if start_idx is None else f'{shape_id}, start_idx: {start_idx:03d}'
            pbar.set_postfix_str(postfix)
            exp_name = shape_id
            exp_args = edict(
                exp_dir=str(root_dir),
                exp_name=exp_name,
                seed=args.seed,
                mesh_file=str(path),
            )
            if start_idx is not None:
                runner.config.camera.start_idx = start_idx
            try:
                runner.run(exp_args, override_existing=False)
            except Exception as e:
                print(f"[WARNING] Error in shape {shape_id}: {e}")
                traceback.print_exc()
                shutil.rmtree(runner.exp_dir)
                print(f"[INFO] Removed experiment directory due to error: {runner.exp_dir}")

        if hasattr(runner, 'start_indices'):
            with open(Path(args.exp_dir) / 'start_indices.txt', 'w') as f:
                for start_idx in runner.start_indices:
                    f.write(f"{start_idx}\n")

if __name__ == '__main__':
    main()