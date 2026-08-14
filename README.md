# A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry

**[Project Page](https://jingsenzhu.github.io/bayesian-nbv-page) | [Paper (arXiv)](https://arxiv.org/abs/2605.05095) | [Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3799902.3811119)**

SIGGRAPH 2026 conference track

## TODOs

- [x] Preliminary code release
- [ ] Release all config files
- [ ] Release the scripts to generate data used by all experiments in the paper
- [ ] Finish the details in [running instructions](#instructions)

## Instructions

### Setup

**TODO:** Will update a complete list in `pyproject.toml` in the future.

Necessary libraries:
- PyTorch with GPU (`torch 2.5.1+cu121` in original experiments)
- `pytorch3d`: May need to manually build and install from source from [their official repo](https://github.com/facebookresearch/pytorch3d) if no pre-compiled wheels match your local environment (PyTorch/CUDA pair)
- `numpy`, `scipy`
- `PyYAML`
- `easydict`, `tqdm`, `matplotlib`, `plotly`
- `trimesh`, `point-cloud-utils`, `gpytoolbox`, `robust-laplacian`

After installing all packages above, run
```
pip install -e .
```

### Quick start

TBD

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

