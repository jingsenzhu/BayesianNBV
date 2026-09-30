# A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry

**[Project Page](https://jingsenzhu.github.io/bayesian-nbv-page) | [Paper (arXiv)](https://arxiv.org/abs/2605.05095) | [Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3799902.3811119)**

SIGGRAPH 2026 conference track

## TODOs

- [x] Preliminary code release (2026/8/14)
- [ ] Release all config files
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

