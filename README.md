# Hessian-Based Molecular Conformation Augmentation for Machine Learning Interatomic Potentials

Demo notebook accompanying:

> Kwak, Bumju, and Jeonghee Jo. "Hessian-based molecular conformation
> augmentation for a scalable and efficient strategy of machine learning
> interatomic potentials." arXiv preprint arXiv:2609.05233 (2026).

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

**Compute:** model training in the paper used a single NVIDIA RTX 3090.
This demo needs only CPU — alanine is small enough (13 atoms, 6-31G*) that a
GPU adds launch/JIT overhead rather than speedup. For larger molecules, swap
`pyscf` for [`gpu4pyscf`](https://github.com/pyscf/gpu4pyscf) to run the same
calls on GPU.

Tested with:

| Package | Version |
|---|---|
| Python | 3.12.13 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |
| pyscf | 2.14.0 |
| rdkit | 2026.03.6 (only for `compute_alanine_qm.py`) |

## Release notes

- **v1.0** (2026-09-11) — initial public release.

## Citation

```bibtex
@article{kwak2026hessian,
  title   = {Hessian-based molecular conformation augmentation for a scalable
             and efficient strategy of machine learning interatomic potentials},
  author  = {Kwak, Bumju and Jo, Jeonghee},
  journal = {arXiv preprint arXiv:2609.05233},
  year    = {2026}
}
```

