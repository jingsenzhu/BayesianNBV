# A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry

**[Project Page](https://jingsenzhu.github.io/bayesian-nbv-page) | [Paper (arXiv)](https://arxiv.org/abs/2605.05095) | [Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3799902.3811119)**

SIGGRAPH 2026 conference track

## TODOs

- [x] Preliminary code release (2026/8/14)
- [ ] Release all config files
  - [x] Template config files ([config/templates/](config/templates))
  - [ ] Config files to reproduce the paper experiments
- [ ] Release the scripts to generate data used by all experiments in the paper
- [ ] Finish the details in [running instructions](#instructions)

## Instructions

### Setup

The original experiments used Python 3.11, `torch 2.5.1+cu121` and `pytorch3d 0.7.8`.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support.
2. Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for your PyTorch/CUDA pair. If no prebuilt wheel matches, build it from source. It is not listed in `pyproject.toml` because it must be built against the installed PyTorch.
3. Install this package and its remaining dependencies:
   ```
   pip install -e .
   ```

### Quick start

**Step 1: precompute prior samples.** Each step of the method conditions posterior samples of the implicit function `f` on the scanned points with pathwise conditioning, which needs draws from the GP prior of `f` evaluated on a 3D grid. These draws do not depend on the object or the scan, so they are generated once by [scripts/precompute_f_samples.py](scripts/precompute_f_samples.py) and reused by all experiments. Run it with a GPU:

```
python scripts/precompute_f_samples.py precompute/l_5e-3_v_0.2
```

The defaults are the settings used in the paper (Matérn-3/2 kernel with lengthscale `0.005` and variance `0.2`, a `100^3` grid of radius `3.5`, truncation `50`, seed `32`; 400 samples in 25 files of 16 samples each). Every setting can be changed from the command line; run the script with `-h` to list them. The script writes `000.npy`, `001.npy`, … (float32, shape `(samples_per_file, grid_density^3)`) and a `config.json` with the kernel and grid parameters to the given directory. Experiments read this directory through `exp.precompute_dir` in the config file, and step `i` consumes the `i`-th group of files, so make sure the directory holds enough files for the number of steps you run.

**Step 2: run an experiment.** Each experiment is configured by YAML files. [config/templates/](config/templates) holds one annotated template per task (every key is commented):

| Task | Driver | Config files |
|---|---|---|
| Classification | [experiments/main_classification.py](experiments/main_classification.py) | `classification/base.yaml` + `classification/classification.yaml` |
| Part segmentation | [experiments/main_partseg.py](experiments/main_partseg.py) | `partseg/base.yaml` + `partseg/partseg.yaml` |
| Heat diffusion | [experiments/main_heat.py](experiments/main_heat.py) | `heat/base.yaml` + `heat/heat.yaml` |
| Reconstruction (Chamfer) | [experiments/main_chamfer.py](experiments/main_chamfer.py) | `chamfer/chamfer.yaml` (one merged file) |

For the first three tasks, the *base* file holds the settings shared across experiments (scan resolution and field of view, SPSR parameters, number of steps, `exp.precompute_dir`). The *task* file holds what is specific to the task, namely the model checkpoint, the task parameters and the `camera` section (candidate views, radius and start view). The driver loads the base file and then replaces whole top-level sections with those of the task file, so a section in the task file must be complete. Copy the templates and edit the paths, such as `exp.precompute_dir` and the checkpoint and class-list paths, before running. Note that `grid_density`, `lengthscale` and the other GP settings are not in the YAML files. They are read from `<precompute_dir>/config.json`.

Below are template commands that run one object with the paper's method (`--exp_type sphere`). Paths in `<...>` are placeholders, and the config paths are relative to the repository root. The first argument is the output directory, and a run writes to `<exp_dir>/<exp_name>/<exp_name>/`. The name is generated from the mesh and a timestamp unless `--exp_name` is given.

```bash
# Classification (the file name must be <class>_<id>.off, e.g. chair_0916.off; the class is the ground truth)
python experiments/main_classification.py exp/cls config/templates/classification/base.yaml config/templates/classification/classification.yaml \
    -mf <path/to/model.off> --exp_type sphere --seed 0

# Part segmentation (the mesh path must have the form .../<category_id>/<shape_id>/models/model_normalized.obj)
python experiments/main_partseg.py exp/partseg config/templates/partseg/base.yaml config/templates/partseg/partseg.yaml \
    -mf <path/to/shapenet/category_id/shape_id/models/model_normalized.obj> --exp_type sphere --seed 0

# Heat diffusion (mesh file: -m)
python experiments/main_heat.py exp/heat config/templates/heat/base.yaml config/templates/heat/heat.yaml \
    -m <path/to/mesh.ply> --exp_type sphere --seed 0

# Reconstruction: a single config
python experiments/main_chamfer.py exp/chamfer config/templates/chamfer/chamfer.yaml \
    -mf <path/to/mesh.off> --exp_type sphere --seed 0
```

Useful options:
- **Other view-selection strategies:** `--exp_type` also accepts the baselines `fps`, `random` and `uncertainty`, and the gradient-descent methods `msgd` and `sphere_msgd`. Part segmentation adds `iterative_sphere_msgd`, and heat diffusion supports only `sphere`, `fps`, `random` and `uncertainty`. The gradient-descent methods need the extra `camera` keys that are commented out in the task templates.
- **Initial view:** `--start_idx` sets the index of the first camera on the candidate sphere and overrides `camera.start_idx`.
- **Batch mode:** pass a mesh list with `-ml <list.txt>` (`-l` for heat) instead of a single mesh, and optionally a per-mesh start-view list with `-sil <start_indices.txt>`. Results go to `<exp_dir>/results/<name>/`, and meshes that already have results are skipped, so an interrupted run can be resumed. The list format depends on the task. Classification and heat read ModelNet shape ids such as `chair_0916` together with `-d <modelnet_dir>` and `-s <split>`. Part segmentation and Chamfer on ShapeNet read full mesh paths. Chamfer reads ModelNet ids when `-dt modelnet` is given.
- **Device:** the code uses the GPU when `torch.cuda.is_available()` and silently falls back to the CPU otherwise. There is no device flag, so check that CUDA is visible before a long run.

The remaining parts of the experiments are TBD.

### Data generation

TBD

## Citation

```
@inproceedings{zhu2026bayesian,
    author={Zhu, Jingsen and Sell{\'a}n, Silvia and Terenin, Alexander},
    title={A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry},
    booktitle={Proceedings of the SIGGRAPH 2026 Conference Papers},
    pages={1--11},
    year={2026}
}
```

