# Hessian-Informed Conformation Augmentation for Potential Energy Surface Curvature in Machine Learning Interatomic Potentials

Demo notebook accompanying:

> Kwak, Bumju, and Jeonghee Jo. "Hessian-informed conformation augmentation
> for potential energy surface curvature in machine learning interatomic
> potentials." arXiv preprint arXiv:2609.05233 (2026).

A small, self-contained demo of two Hessian-based data augmentation methods
for training machine-learned interatomic potentials (MLIPs), shown on one
real molecule (alanine, 13 atoms).

Both methods synthesize a displaced structure and label it with a
second-order Taylor expansion around the ground-truth minimum:

$$\Delta E = -\mathbf{F}^{\top}\delta\mathbf{x} + \tfrac12\delta\mathbf{x}^{\top}\mathbf{H}\delta\mathbf{x}, \qquad \Delta\mathbf{F} = -\mathbf{H}\delta\mathbf{x}$$

- **UniAug** — random atom subset, isotropic Gaussian noise (curvature-agnostic)
- **ModeAug** — all atoms, weighted sum over normal modes, $w_k\propto\lambda_k^{-1/2}$ (curvature-aware)

They differ only in how $\delta\mathbf{x}$ is drawn.

## Contents

| File | What it is |
|---|---|
| [`uniaug_modeaug_intro.ipynb`](uniaug_modeaug_intro.ipynb) | The demo: visual comparison of both methods, plus a live DFT cross-check |
| `alanine_qm_data.npz` | Precomputed RHF/6-31G* geometry, Hessian, and 33 real normal modes for alanine |
| `compute_alanine_qm.py` | Script that generated the `.npz` (RDKit embed → MMFF → RHF/6-31G* optimization → analytic Hessian, via PySCF) |
| [`train_direct_force_equiformerV2.py`](train_direct_force_equiformerV2.py) | EquiformerV2 training (direct force) |
| [`train_autograd_force_gotennet.py`](train_autograd_force_gotennet.py) | GotenNet training (autograd force) |
| [`ts1x_utils.py`](ts1x_utils.py) | Shared utilities (dataset, UniAug / ModeAug, normalization) |

## Running the notebook

```
pip install numpy matplotlib pyscf
jupyter notebook uniaug_modeaug_intro.ipynb
```

- Sections 1–6 (visualization) need only `numpy` + `matplotlib` — the
  geometry/Hessian are already cached in the `.npz`.
- Section 7 (real DFT cross-check) additionally needs `pyscf` and runs a
  live RHF/6-31G* Hessian + ~32 single-point calculations (a few minutes on
  CPU).

To regenerate `alanine_qm_data.npz` from scratch, or swap in a different
molecule, run `compute_alanine_qm.py` (also needs `rdkit`).

**Compute**

- Model training: a single NVIDIA RTX 3090 or RTX 4090
- Demo notebook: CPU only (alanine is small enough that a GPU adds overhead)
- Larger molecules: swap `pyscf` for [`gpu4pyscf`](https://github.com/pyscf/gpu4pyscf)

Tested with:

| Package | Version |
|---|---|
| Python | 3.12.13 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |
| pyscf | 2.14.0 |
| rdkit | 2026.03.6 (only for `compute_alanine_qm.py`) |

Training environment:

| Package | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.2.1+cu121 (torchvision 0.17.1, torchaudio 2.2.1) |
| torch-geometric | 2.7.0 |
| torch_scatter / torch_sparse / torch_cluster | 2.1.2 / 0.6.18 / 1.6.3 (`+pt22cu121`) |
| numpy | 1.26.4 (pinned) |

## Training scripts

- Train on Transition1x with optional on-the-fly UniAug / ModeAug (`--augment`, needs a training LMDB with Hessians)
- Model packages are not included; set the path or install:
  - EquiformerV2 → `EQUIFORMER_V2_PATH`
  - [GotenNet](https://github.com/sarpaykent/GotenNet) → `GOTENNET_PATH`
- Key options
  - `--aug_method {uniaug,modeaug}`, `--aug_sigma`, `--aug_atom_frac` (UniAug), `--aug_multiplier`, `--aug_only`
  - `--init_mode` (EquiformerV2: OC20 31M/83M, RGD1, scratch, HORM cfg; GotenNet: scratch only)
  - `--best_metric {direct,autograd}` (GotenNet: autograd only)
  - `--resume_ckpt` (EquiformerV2) / `--resume_from` (GotenNet)
- Normalization
  - Per-atom energy with linear reference (JMP / HORM recipe)
  - Forces: `f_std` (direct), `e_std` (autograd)
- Outputs in `--run_dir`: args, norm stats, linref, CSV logs, best / final checkpoints

```
python train_direct_force_equiformerV2.py --augment --aug_method modeaug --aug_sigma 0.05
python train_autograd_force_gotennet.py   --augment --aug_method uniaug  --aug_sigma 0.05
```

## Release notes

- **v1.1** (2026-10-13) — added training scripts and shared utilities.
- **v1.0** (2026-09-11) — initial public release.

## Citation

```bibtex
@article{kwak2026hessian,
  title   = {Hessian-informed conformation augmentation for potential energy
             surface curvature in machine learning interatomic potentials},
  author  = {Kwak, Bumju and Jo, Jeonghee},
  journal = {arXiv preprint arXiv:2609.05233},
  year    = {2026}
}
```

