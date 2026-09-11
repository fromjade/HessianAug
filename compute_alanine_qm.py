#!/usr/bin/env python
"""Compute a REAL geometry + Hessian + normal modes for alanine.

This follows the "make it real" recipe from ../fig1a_formaldehyde_figure_README.md:

    1. Take a real equilibrium geometry (RDKit ETKDG embed + MMFF pre-opt,
       then a genuine QM optimization).
    2. Compute the Hessian at that geometry (RHF/6-31G* analytic Hessian, PySCF).
    3. Diagonalize the mass-weighted Hessian for true normal modes + eigenvalues
       (via pyscf.hessian.thermo.harmonic_analysis, which also projects out the
       6 zero translation/rotation modes).
    4. Save everything so make_fig_alanine.py can draw actual eigenvectors.

Nothing in this script is hand-placed: the geometry is optimized, the Hessian
is analytic, and the 33 = 3*13-6 reported frequencies are the genuine harmonic
normal-mode frequencies of neutral gas-phase alanine at this level of theory.

    python compute_alanine_qm.py   # writes alanine_qm_data.npz (~seconds-minutes)
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf
from pyscf.geomopt.berny_solver import optimize
from pyscf.hessian import thermo

SMILES = "CC(N)C(=O)O"   # neutral (non-zwitterionic) gas-phase alanine
BASIS = "6-31g*"
SEED = 42


def build_initial_geometry():
    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    AllChem.EmbedMolecule(mol, randomSeed=SEED, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
              2 if b.GetBondTypeAsDouble() == 2.0 else 1)
             for b in mol.GetBonds()]
    return symbols, coords, bonds


def main():
    symbols, coords0, bonds = build_initial_geometry()
    atom_str = "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, coords0))

    mol = gto.M(atom=atom_str, basis=BASIS, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print("initial RHF energy:", mf.e_tot)

    mol_eq = optimize(mf, maxsteps=100)
    mf_eq = scf.RHF(mol_eq)
    e_eq = mf_eq.kernel()
    print("optimized RHF energy:", e_eq)

    hess = mf_eq.Hessian().kernel()               # (natm, natm, 3, 3), analytic
    info = thermo.harmonic_analysis(mol_eq, hess)  # projects out translation/rotation

    coords_eq = mol_eq.atom_coords(unit="Angstrom")
    freq = info["freq_wavenumber"]                 # (3N-6,) real cm^-1, ascending
    modes = info["norm_mode"]                       # (3N-6, natm, 3) Cartesian eigenvectors
    assert np.all(np.isreal(freq)) and np.all(freq > 0), "found an imaginary mode: not a true minimum"

    np.savez("alanine_qm_data.npz",
             symbols=np.array(symbols), coords_ang=coords_eq, bonds=np.array(bonds),
             freq_wavenumber=freq, norm_mode=modes, energy_hartree=e_eq, basis=BASIS)
    print(f"wrote alanine_qm_data.npz  ({len(symbols)} atoms, {len(freq)} real vibrational modes, "
          f"{freq[0]:.1f}-{freq[-1]:.1f} cm^-1)")


if __name__ == "__main__":
    main()
