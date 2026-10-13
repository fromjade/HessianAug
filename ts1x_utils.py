"""
Shared utilities for the Transition1x training scripts.

Contains the LMDB dataset, the collate function, the Hessian based UniAug and
ModeAug augmentations, linear reference and normalization statistics, the
learning rate schedule, and seeding. Both train_direct_force_equiformerV2.py
and train_autograd_force_gotennet.py import from this module.
"""

import pickle
import random

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_scatter import scatter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONE_HOT_TO_ATOMIC_NUM = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)
_ATOMIC_MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998}
_TR_THRESHOLD = 1e-2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Ts1xDataset(Dataset):
    """LMDB dataset with lazy env opening (multi worker safe).

    Handles several LMDB formats (one_hot, raw charges, or atomic_numbers) and
    exposes pos, atomic_numbers, natoms, energy, forces (and hessian if stored)
    on a PyG Data object.
    """

    def __init__(self, lmdb_path):
        self.lmdb_path = str(lmdb_path)
        self._env = None

        env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False,
                        readahead=False, meminit=False, max_readers=1)
        with env.begin() as txn:
            self.num_samples = pickle.loads(txn.get("length".encode("ascii")))
        env.close()

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=1,
            )
        return self._env

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        env = self._get_env()
        with env.begin() as txn:
            raw = pickle.loads(txn.get(f"{idx}".encode("ascii")))

        if hasattr(raw, "charges"):
            atomic_numbers = raw.charges.long()
        elif hasattr(raw, "one_hot"):
            atomic_numbers = ONE_HOT_TO_ATOMIC_NUM[raw.one_hot.argmax(dim=-1)].long()
        elif hasattr(raw, "atomic_numbers"):
            atomic_numbers = raw.atomic_numbers.long()
        else:
            raise ValueError(f"No atomic number info in sample {idx}")

        natoms = raw.pos.shape[0]
        energy_val = (raw.energy.float() if isinstance(raw.energy, torch.Tensor)
                      else torch.tensor(raw.energy, dtype=torch.float))
        data = Data(
            pos=raw.pos.float(),
            atomic_numbers=atomic_numbers,
            natoms=torch.tensor([natoms], dtype=torch.long),
            energy=energy_val.reshape(1),
            forces=raw.forces.float(),
        )
        if hasattr(raw, "hessian") and raw.hessian is not None:
            data.hessian = raw.hessian.float()

        return data


def collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    # Ensure natoms is 1D with shape [B]
    batch.natoms = torch.tensor([int(d.natoms.item()) for d in data_list],
                                dtype=torch.long)
    return batch


# ---------------------------------------------------------------------------
# Hessian based augmentation (second order Taylor expansion)
# ---------------------------------------------------------------------------

def uniaug(data, sigma=0.05, atom_frac=0.25):
    """UniAug. Uniform random Cartesian displacement on a random subset of atoms.

    Energy and forces of the displaced geometry are obtained from the stored
    forces and Hessian through a second order Taylor expansion.
    """
    natoms = int(data.natoms.item())
    if not hasattr(data, "hessian") or data.hessian is None:
        return data

    hessian = data.hessian.reshape(natoms * 3, natoms * 3).float()
    forces = data.forces.reshape(-1).float()
    pos = data.pos.float()
    energy = float(data.energy)

    n_displace = max(1, int(natoms * atom_frac))
    atom_indices = random.sample(range(natoms), n_displace)

    dx = torch.zeros(natoms * 3)
    for idx in atom_indices:
        dx[idx * 3:(idx + 1) * 3] = torch.randn(3) * sigma

    dE = -forces.dot(dx) + 0.5 * dx @ hessian @ dx
    dF = -(hessian @ dx)

    return Data(
        pos=pos + dx.reshape(natoms, 3),
        atomic_numbers=data.atomic_numbers.clone(),
        natoms=torch.tensor([natoms], dtype=torch.long),
        energy=torch.tensor([energy + dE.item()], dtype=torch.float),
        forces=(forces + dF).reshape(natoms, 3),
    )


def modeaug(data, sigma=0.05):
    """ModeAug. Eigenvalue weighted displacement in normal mode space.

    The mass weighted Hessian is diagonalized and each mode is excited with a
    random coefficient scaled by the inverse square root of its eigenvalue.
    Translation and rotation modes (near zero eigenvalues) are excluded.
    """
    natoms = int(data.natoms.item())
    if not hasattr(data, "hessian") or data.hessian is None:
        return data

    hessian = data.hessian.reshape(natoms * 3, natoms * 3).double().numpy()
    forces = data.forces.reshape(-1).double().numpy()
    pos = data.pos.double().numpy()
    energy = float(data.energy)

    atomic_numbers = data.atomic_numbers.long().tolist()
    masses = np.array([_ATOMIC_MASSES[z] for z in atomic_numbers])
    mass_3n = np.repeat(masses, 3)
    inv_sqrt_mass = 1.0 / np.sqrt(mass_3n)

    H_mw = hessian * np.outer(inv_sqrt_mass, inv_sqrt_mass)
    eigenvalues, eigenvectors = np.linalg.eigh(H_mw)

    include_mask = np.abs(eigenvalues) >= _TR_THRESHOLD
    n_included = int(include_mask.sum())
    if n_included == 0:
        return data

    weights = np.zeros(3 * natoms)
    weights[include_mask] = 1.0 / np.sqrt(np.abs(eigenvalues[include_mask]))
    weight_norm = np.linalg.norm(weights[include_mask])
    weights[include_mask] = weights[include_mask] / weight_norm * np.sqrt(n_included)

    coeffs = np.zeros(3 * natoms)
    coeffs[include_mask] = np.random.randn(n_included) * weights[include_mask] * sigma

    dx = inv_sqrt_mass * (eigenvectors @ coeffs)

    dE = -forces @ dx + 0.5 * dx @ hessian @ dx
    dF = -(hessian @ dx)

    return Data(
        pos=torch.from_numpy(pos + dx.reshape(natoms, 3)).float(),
        atomic_numbers=data.atomic_numbers.clone(),
        natoms=torch.tensor([natoms], dtype=torch.long),
        energy=torch.tensor([energy + dE], dtype=torch.float),
        forces=torch.from_numpy((forces + dF).reshape(natoms, 3)).float(),
    )


# ---------------------------------------------------------------------------
# Linear reference and normalization
# ---------------------------------------------------------------------------

def compute_linref_coeffs(dataset, max_z=10, max_samples=None):
    """Per element linear reference energies via least squares on the training set."""
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    indices = (random.sample(range(len(dataset)), n)
               if max_samples is not None else list(range(n)))

    X = np.zeros((n, max_z), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for i, idx in enumerate(indices):
        data = dataset[idx]
        for z in data.atomic_numbers.long().numpy():
            if z < max_z:
                X[i, z] += 1
        y[i] = float(data.energy)

    used_mask = X.sum(axis=0) > 0
    X_used = X[:, used_mask]
    coeff_used, _, _, _ = np.linalg.lstsq(X_used, y, rcond=None)

    coeff = np.zeros(max_z, dtype=np.float64)
    coeff[used_mask] = coeff_used

    z_names = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    print("  Linear reference coefficients (eV)")
    for z in range(max_z):
        if used_mask[z]:
            print(f"    {z_names.get(z, f'Z={z}')} (Z={z}) = {coeff[z]:.4f}")

    return coeff


def compute_linref_energy_batch(batch, linref_tensor):
    per_atom_linref = linref_tensor[batch.atomic_numbers.long()]
    return scatter(per_atom_linref, batch.batch, dim=0, reduce="sum")


def compute_normalization_stats(dataset, linref=None, max_samples=None):
    """Per atom energy mean and std, and force std.

    max_samples=None uses every sample in the dataset (deterministic).
    """
    if max_samples is None or max_samples >= len(dataset):
        sample_indices = range(len(dataset))
    else:
        sample_indices = random.sample(range(len(dataset)), max_samples)

    energies = []
    all_forces = []
    for idx in sample_indices:
        data = dataset[idx]
        nat = int(data.natoms.item())
        e_total = float(data.energy)
        if linref is not None:
            for z in data.atomic_numbers.long().numpy():
                e_total -= linref[z]
        energies.append(e_total / nat)
        all_forces.append(data.forces.float().reshape(-1))

    return {
        "energy_mean_per_atom": float(np.mean(energies)),
        "energy_std_per_atom": float(max(np.std(energies), 1e-6)),
        "force_std": float(max(torch.cat(all_forces).std().item(), 1e-6)),
    }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_factor=0.01):
    """Linear warmup followed by cosine decay, step based."""
    def lr_lambda(step):
        if step < warmup_steps:
            return min_factor + (1.0 - min_factor) * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1 - min_factor) * 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)