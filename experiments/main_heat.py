import argparse
import shutil
from pathlib import Path

import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from bayesian_nbv.exp_runner.fps import FPSRunner
from bayesian_nbv.exp_runner.heat_diffusion import HeatDiffusionRunner
from bayesian_nbv.exp_runner.random import RandomRunner
from bayesian_nbv.exp_runner.sphere_search_iterative import SphereSearchIterativeRunner
from bayesian_nbv.exp_runner.uncertainty import RayUncertaintyRunner


class HeatDiffusionSphereSearchRunner(SphereSearchIterativeRunner, HeatDiffusionRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)

    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)


class HeatDiffusionFPSRunner(FPSRunner, HeatDiffusionRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)

    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)


class HeatDiffusionRandomRunner(RandomRunner, HeatDiffusionRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)

    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)


class HeatDiffusionRayUncertaintyRunner(RayUncertaintyRunner, HeatDiffusionRunner):
    def __init__(self, config: edict, verbose: bool = True):
        super().__init__(config, verbose)

    def init_step_checkpoint(self, **kwargs):
        return super().init_step_checkpoint(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Path to the experiment directory')
    parser.add_argument('base_config', type=str, help='Path to the base config file')
    parser.add_argument('heat_diffusion_config', type=str, help='Path to the heat diffusion config file')
    parser.add_argument('-m', '--mesh_file', type=str, default=None, help='Path to the mesh file')
    parser.add_argument('-l', '--mesh_list', type=str, default=None, help='Path to the mesh list file')
    parser.add_argument('-d', '--mesh_dir', type=str, default=None, help='Path to the mesh directory')
    parser.add_argument('-s', '--split', type=str, choices=['train', 'test'], default='train', help='Dataset split to use')
    parser.add_argument('-sil', '--start_idx_list', type=str, default=None, help='Start indices list file for selection')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Whether to run in quiet mode')
    parser.add_argument('--exp_type', type=str, choices=['sphere', 'fps', 'random', 'uncertainty'], default='sphere', help='Experiment type')
    args = parser.parse_args()

    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    config = edict(base_config)
    with open(args.heat_diffusion_config, 'r') as f:
        heat_diffusion_config = yaml.safe_load(f)
    config.update(heat_diffusion_config)

    assert args.mesh_file is not None or args.mesh_list is not None, "Either mesh_file or mesh_list must be provided"

    runner_classes = {
        'sphere': HeatDiffusionSphereSearchRunner,
        'fps': HeatDiffusionFPSRunner,
        'random': HeatDiffusionRandomRunner,
        'uncertainty': HeatDiffusionRayUncertaintyRunner,
    }
    runner = runner_classes[args.exp_type](config, verbose=not args.quiet)

    if args.mesh_file is not None:
        exp_args = edict(
            exp_dir=args.exp_dir,
            exp_name=args.exp_name,
            seed=args.seed,
            mesh_file=args.mesh_file,
        )
        runner.run(exp_args)
    else:
        assert args.mesh_dir is not None, "mesh_dir must be provided"
        runner.verbose = False
        root_dir = Path(args.exp_dir) / 'results'
        root_dir.mkdir(parents=True, exist_ok=True)
        with open(args.mesh_list, 'r') as f:
            mesh_list = f.readlines()
        mesh_list = [line.strip() for line in mesh_list]
        shape_ids = mesh_list
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
        datapath = [
            str(Path(args.mesh_dir) / shape_names[i] / args.split / (shape_ids[i] + '.off'))
            for i in range(len(shape_ids))
        ]

        if args.start_idx_list is not None:
            with open(args.start_idx_list, 'r') as f:
                start_idx_list = f.readlines()
            start_idx_list = [int(line.strip()) for line in start_idx_list]
            assert len(start_idx_list) == len(shape_ids), "Number of start indices must be equal to number of test meshes"
        else:
            start_idx_list = [None] * len(shape_ids)

        pbar = tqdm(zip(shape_ids, datapath, start_idx_list), desc="Processing meshes", total=len(datapath))
        for shape_id, path, start_idx in pbar:
            postfix = shape_id if start_idx is None else f'{shape_id}, start_idx: {start_idx:03d}'
            pbar.set_postfix_str(postfix)
            exp_args = edict(
                exp_dir=str(root_dir),
                exp_name=shape_id,
                seed=args.seed,
                mesh_file=path,
            )
            if start_idx is not None:
                runner.config.camera.start_idx = start_idx
            try:
                runner.run(exp_args, override_existing=False)
            except Exception as e:
                print(f"[WARNING] Error in shape {shape_id}: {e}")
                shutil.rmtree(runner.exp_dir)
                print(f"[INFO] Removed experiment directory due to error: {runner.exp_dir}")


if __name__ == '__main__':
    main()
