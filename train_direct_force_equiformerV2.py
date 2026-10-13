"""
EquiformerV2 training on Transition1x (JMP v4 style).

Trains EquiformerV2 with a direct force head. Energy and force targets are
normalized as in the JMP and HORM recipes. Validation reports both the direct
force and the autograd force (negative gradient of the predicted energy). The
best checkpoint is selected with --best_metric, which defaults to the direct
force that the model is trained on.

Initialization options
  oc20_31m, oc20_83m   OC20 pretrained EquiformerV2 checkpoints (paths via
                       --oc20_31m_ckpt and --oc20_83m_ckpt)
  rgd1                 a checkpoint trained on RGD1 (path via --rgd1_ckpt_path)
  scratch, horm_cfg, horm_cfg_3l   training from scratch

The EquiformerV2 package is not included here. Clone the official repository
and point the EQUIFORMER_V2_PATH environment variable to it, or run this script
from the repository root. Shared data and normalization utilities live in
ts1x_utils.py.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_EQUIFORMER_V2_PATH = os.environ.get("EQUIFORMER_V2_PATH")
if _EQUIFORMER_V2_PATH:
    sys.path.insert(0, _EQUIFORMER_V2_PATH)
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from ts1x_utils import (
    Ts1xDataset,
    collate_fn,
    compute_linref_coeffs,
    compute_linref_energy_batch,
    compute_normalization_stats,
    get_cosine_schedule_with_warmup,
    modeaug,
    set_seed,
    uniaug,
)


# ---------------------------------------------------------------------------
# Model build + pretrained loading
# ---------------------------------------------------------------------------

# Architecture configs matching the OC20 yml files of the official EquiformerV2
# repository so that
# pretrained state_dicts load cleanly. The `scratch` variant mirrors the 31M
# arch but shrinks max_num_elements to 10 (T1X only covers up to Z=9).
MODEL_CONFIGS = {
    "oc20_31m": dict(
        num_layers=8, sphere_channels=128,
        attn_hidden_channels=64, num_heads=8,
        attn_alpha_channels=64, attn_value_channels=16,
        ffn_hidden_channels=128,
        lmax_list=[4], mmax_list=[2], grid_resolution=18,
        edge_channels=128, num_distance_basis=512,
        alpha_drop=0.1, drop_path_rate=0.1, proj_drop=0.0,
        max_num_elements=90,
    ),
    "oc20_83m": dict(
        num_layers=12, sphere_channels=128,
        attn_hidden_channels=64, num_heads=8,
        attn_alpha_channels=64, attn_value_channels=16,
        ffn_hidden_channels=128,
        lmax_list=[6], mmax_list=[2], grid_resolution=18,
        edge_channels=128, num_distance_basis=512,
        alpha_drop=0.1, drop_path_rate=0.05, proj_drop=0.0,
        max_num_elements=90,
    ),
    "scratch": dict(
        num_layers=8, sphere_channels=128,
        attn_hidden_channels=64, num_heads=8,
        attn_alpha_channels=64, attn_value_channels=16,
        ffn_hidden_channels=128,
        lmax_list=[4], mmax_list=[2], grid_resolution=18,
        edge_channels=128, num_distance_basis=512,
        alpha_drop=0.1, drop_path_rate=0.1, proj_drop=0.0,
        max_num_elements=10,
    ),
    # EquiformerV2 architecture used in the HORM paper. Use --init_mode horm_cfg
    # for a direct comparison with the HORM results.
    "horm_cfg": dict(
        num_layers=4, sphere_channels=128,
        attn_hidden_channels=64, num_heads=4,
        attn_alpha_channels=64, attn_value_channels=16,
        ffn_hidden_channels=128,
        lmax_list=[4], mmax_list=[2], grid_resolution=18,
        edge_channels=128, num_distance_basis=512,
        alpha_drop=0.1, drop_path_rate=0.1, proj_drop=0.0,
        max_num_elements=90,
    ),
    # horm_cfg with num_layers=3 (smaller variant for speed / memory ablation).
    "horm_cfg_3l": dict(
        num_layers=3, sphere_channels=128,
        attn_hidden_channels=64, num_heads=4,
        attn_alpha_channels=64, attn_value_channels=16,
        ffn_hidden_channels=128,
        lmax_list=[4], mmax_list=[2], grid_resolution=18,
        edge_channels=128, num_distance_basis=512,
        alpha_drop=0.1, drop_path_rate=0.1, proj_drop=0.0,
        max_num_elements=90,
    ),
}


def build_model(cfg, max_radius, max_neighbors):
    return EquiformerV2_OC20(
        num_atoms=None, bond_feat_dim=None, num_targets=None,
        use_pbc=False, regress_forces=True, otf_graph=True,
        max_neighbors=max_neighbors, max_radius=max_radius,
        max_num_elements=cfg["max_num_elements"],
        num_layers=cfg["num_layers"],
        sphere_channels=cfg["sphere_channels"],
        attn_hidden_channels=cfg["attn_hidden_channels"],
        num_heads=cfg["num_heads"],
        attn_alpha_channels=cfg["attn_alpha_channels"],
        attn_value_channels=cfg["attn_value_channels"],
        ffn_hidden_channels=cfg["ffn_hidden_channels"],
        norm_type="layer_norm_sh",
        lmax_list=cfg["lmax_list"],
        mmax_list=cfg["mmax_list"],
        grid_resolution=cfg["grid_resolution"],
        num_sphere_samples=128,
        edge_channels=cfg["edge_channels"],
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=cfg["num_distance_basis"],
        attn_activation="silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="silu",
        use_gate_act=False,
        use_grid_mlp=True,
        use_sep_s2_act=True,
        alpha_drop=cfg["alpha_drop"],
        drop_path_rate=cfg["drop_path_rate"],
        proj_drop=cfg["proj_drop"],
        weight_init="uniform",
    )


def _strip_module_prefix(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module.module."):
            nk = nk[len("module.module."):]
        elif nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out


def load_pretrained(model, ckpt_path, init_mode):
    """Load a pretrained checkpoint (OC20 or RGD1-trained) into `model`.
    Uses strict=False; returns (missing, unexpected) for logging."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if init_mode in ("oc20_31m", "oc20_83m"):
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        sd = _strip_module_prefix(sd)
    elif init_mode == "rgd1":
        sd = (ckpt["model_state_dict"] if isinstance(ckpt, dict)
              and "model_state_dict" in ckpt else ckpt)
        sd = _strip_module_prefix(sd)
    else:
        raise ValueError(f"load_pretrained called with init_mode={init_mode}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, scheduler, device, args, norm_stats, linref_tensor,
    epoch, dataset=None, batch_log_file=None, global_step=0,
    val_loader=None, best_val_f_mae=float("inf"), timestamp="",
    step_val_log_file=None,
):
    model.train()
    total_loss = 0.0
    total_e_mae = 0.0
    total_f_mae = 0.0
    n_batches = 0

    e_mean = norm_stats["energy_mean_per_atom"]
    e_std = norm_stats["energy_std_per_atom"]
    f_std = norm_stats["force_std"]
    use_linref = linref_tensor is not None

    for batch_idx, batch in enumerate(loader):
        # --- Augmentation ---
        if args.augment and dataset is not None:
            aug_list = []
            n_aug = batch.num_graphs * args.aug_multiplier
            for _ in range(n_aug):
                src_idx = random.randint(0, len(dataset) - 1)
                src_data = dataset[src_idx]
                sigma = random.uniform(args.aug_sigma_min, args.aug_sigma_max)
                if args.aug_method == "modeaug":
                    aug_data = modeaug(src_data, sigma=sigma)
                else:
                    aug_data = uniaug(src_data, sigma=sigma,
                                              atom_frac=args.aug_atom_frac)
                aug_list.append(aug_data)

            if args.aug_only:
                batch = collate_fn(aug_list)
            else:
                orig_list = batch.to_data_list()
                for d in orig_list:
                    if hasattr(d, "hessian"):
                        del d.hessian
                batch = collate_fn(orig_list + aug_list)

        batch = batch.to(device)

        natoms_per_mol = batch.natoms
        if natoms_per_mol.dim() == 0:
            natoms_per_mol = natoms_per_mol.unsqueeze(0)

        target_e = batch.energy.float()
        target_f = batch.forces.float()

        if use_linref:
            linref_e = compute_linref_energy_batch(batch, linref_tensor)
            target_e_for_norm = target_e - linref_e
        else:
            target_e_for_norm = target_e

        # JMP v4 convention: per-atom energy normalization; model learns in normalized space.
        target_e_norm = (target_e_for_norm / natoms_per_mol.float() - e_mean) / e_std
        target_f_norm = target_f / f_std

        energy_pred, force_pred = model(batch)
        energy_pred_norm = energy_pred / natoms_per_mol.float()

        # Energy: L1 on per-atom normalized energy.
        e_loss = nn.functional.l1_loss(energy_pred_norm, target_e_norm)
        # Force: per-atom L2 norm of (pred − target_f_norm), averaged (JMP/HORM).
        f_loss = nn.functional.pairwise_distance(
            force_pred, target_f_norm, p=2.0
        ).mean()
        loss = args.energy_coeff * e_loss + args.force_coeff * f_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # --- Logging in physical units ---
        e_mae_phys = e_loss.item() * e_std             # eV/atom
        f_mae_phys = f_loss.item() * f_std             # eV/Å (per-atom L2 norm)
        loss_val = loss.item()

        total_loss += loss_val
        total_e_mae += e_mae_phys
        total_f_mae += f_mae_phys
        n_batches += 1
        global_step += 1

        if batch_log_file is not None:
            lr = optimizer.param_groups[0]["lr"]
            batch_log_file.write(
                f"{global_step},{epoch},{batch_idx+1},"
                f"{loss_val:.6f},{e_mae_phys:.6f},{f_mae_phys:.6f},{lr:.2e}\n"
            )
            if global_step % args.log_every == 0:
                batch_log_file.flush()

        if global_step % args.log_every == 0:
            print(f"  [Step {global_step}] Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss={loss_val:.4f} E_MAE={e_mae_phys:.4f} eV/atom "
                  f"F_MAE={f_mae_phys:.4f} eV/Å")

        # --- Step-based validation ---
        if (args.eval_every_steps > 0
                and global_step % args.eval_every_steps == 0
                and val_loader is not None):
            val_metrics = evaluate(model, val_loader, device, norm_stats, linref_tensor)
            val_e = val_metrics["e_mae"]
            val_fd = val_metrics["f_mae_direct"]
            val_fa = val_metrics["f_mae_autograd"]
            lr = optimizer.param_groups[0]["lr"]
            print(f"  >> [Step {global_step}] Val E_MAE={val_e:.4f} eV/atom | "
                  f"F_direct={val_fd:.4f} | F_autograd={val_fa:.4f} eV/Å")

            if step_val_log_file is not None:
                step_val_log_file.write(
                    f"{global_step},{epoch},{val_e:.6f},"
                    f"{val_fd:.6f},{val_fa:.6f},{lr:.2e}\n"
                )
                step_val_log_file.flush()

            val_f_selected = val_fa if args.best_metric == "autograd" else val_fd
            if args.save_best and val_f_selected < best_val_f_mae:
                best_val_f_mae = val_f_selected
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_e_mae": val_e,
                    "val_f_mae_direct": val_fd,
                    "val_f_mae_autograd": val_fa,
                    "norm_stats": norm_stats,
                    "linref": (linref_tensor.cpu().numpy().tolist()
                               if linref_tensor is not None else None),
                    "args": vars(args),
                }, os.path.join(args.run_dir, f"best_model_{timestamp}.pt"))
                print(f"     -> Best model (F_{args.best_metric}={val_f_selected:.4f})")

            model.train()

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    return {
        "loss": total_loss / max(n_batches, 1),
        "e_mae": total_e_mae / max(n_batches, 1),
        "f_mae": total_f_mae / max(n_batches, 1),
        "global_step": global_step,
        "best_val_f_mae": best_val_f_mae,
    }


def evaluate(model, loader, device, norm_stats, linref_tensor):
    """Single forward per batch → direct-f + autograd, all in physical units.

    Returns:
      e_mae:          per-atom energy MAE (eV/atom)
      f_mae_direct:   direct-f force MAE (eV/Å)  = mean |force_pred*f_std − target|
      f_mae_autograd: autograd force MAE (eV/Å) = mean |(-dE/dr)*e_std − target|
    """
    model.eval()

    e_mean = norm_stats["energy_mean_per_atom"]
    e_std = norm_stats["energy_std_per_atom"]
    f_std = norm_stats["force_std"]
    use_linref = linref_tensor is not None

    all_e_errors = []
    all_f_errors_direct = []
    all_f_errors_autograd = []

    for batch in loader:
        batch = batch.to(device)

        natoms_per_mol = batch.natoms
        if natoms_per_mol.dim() == 0:
            natoms_per_mol = natoms_per_mol.unsqueeze(0)

        target_e = batch.energy.float()
        target_f = batch.forces.float()

        with torch.enable_grad():
            pos = batch.pos.detach().clone().requires_grad_(True)
            batch.pos = pos
            energy_pred, force_pred_direct = model(batch)
            force_pred_autograd = -torch.autograd.grad(
                energy_pred.sum(), pos, create_graph=False,
            )[0]

        # Direct-f: model output in normalized space → * f_std
        pred_f_direct = (force_pred_direct * f_std).detach()
        # Autograd: d/dr of (E_pred * e_std + e_mean*N + linref) = e_std * d(E_pred)/dr
        pred_f_autograd = (force_pred_autograd * e_std).detach()

        all_f_errors_direct.append((pred_f_direct - target_f).abs().mean(dim=-1))
        all_f_errors_autograd.append((pred_f_autograd - target_f).abs().mean(dim=-1))

        pred_e_per_atom = energy_pred.detach() / natoms_per_mol.float()
        pred_e_unnorm = (pred_e_per_atom * e_std + e_mean) * natoms_per_mol.float()
        if use_linref:
            linref_e = compute_linref_energy_batch(batch, linref_tensor)
            pred_e_total = pred_e_unnorm + linref_e
        else:
            pred_e_total = pred_e_unnorm

        e_err = (pred_e_total / natoms_per_mol.float()
                 - target_e / natoms_per_mol.float()).abs()
        all_e_errors.append(e_err)

    return {
        "e_mae": torch.cat(all_e_errors).mean().item(),
        "f_mae_direct": torch.cat(all_f_errors_direct).mean().item(),
        "f_mae_autograd": torch.cat(all_f_errors_autograd).mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EquiformerV2 fine-tuning on Transition-1x (JMP v4 style)")

    # Data
    parser.add_argument("--train_lmdb", type=str,
                        default="data/ts1x_hess_train.lmdb")
    parser.add_argument("--val_lmdb", type=str,
                        default="data/ts1x-val.lmdb")

    # Model / init
    parser.add_argument("--init_mode", type=str, default="oc20_31m",
                        choices=["oc20_31m", "oc20_83m", "rgd1", "scratch", "horm_cfg", "horm_cfg_3l"])
    parser.add_argument("--oc20_31m_ckpt", type=str,
                        default="checkpoints/eq2_31M_ec4_allmd.pt")
    parser.add_argument("--oc20_83m_ckpt", type=str,
                        default="checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--rgd1_ckpt_path", type=str, default=None,
                        help="Required if --init_mode=rgd1")
    parser.add_argument("--max_radius", type=float, default=6.0)
    parser.add_argument("--max_neighbors", type=int, default=50)

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=10.0)
    parser.add_argument("--energy_coeff", type=float, default=2.0)
    parser.add_argument("--force_coeff", type=float, default=100.0)
    parser.add_argument("--head_lr_multiplier", type=float, default=5.0,
                        help="LR multiplier for energy_block/force_block vs backbone")
    parser.add_argument("--best_metric", type=str, default="direct",
                        choices=["autograd", "direct"],
                        help="Validation force MAE used to select the best checkpoint. "
                             "The model is trained with the direct force head, so "
                             "direct is the default.")
    parser.add_argument("--warmup_steps", type=int, default=50000)
    parser.add_argument("--lr_min_factor", type=float, default=0.02)

    # Linear reference
    parser.add_argument("--linref", action="store_true", default=True,
                        help="Subtract per-element linref energies (default ON)")
    parser.add_argument("--no_linref", action="store_true", default=False)
    parser.add_argument("--linref_path", type=str, default=None,
                        help="Path to cached linref_*.npz; if set, skip fitting.")
    parser.add_argument("--linref_max_z", type=int, default=10)
    parser.add_argument("--linref_max_samples", type=int, default=None)
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Path to cached norm_stats_*.json; if set, skip computation.")
    parser.add_argument("--norm_stats_max_samples", type=int, default=None,
                        help="None → use the entire train set (default, deterministic). "
                             "Set an int to randomly subsample.")

    # Augmentation
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--aug_only", action="store_true", default=False)
    parser.add_argument("--aug_method", type=str, default="uniaug",
                        choices=["uniaug", "modeaug"])
    parser.add_argument("--aug_sigma", type=float, default=None,
                        help="Fixed sigma (shortcut for min=max).")
    parser.add_argument("--aug_sigma_min", type=float, default=0.05)
    parser.add_argument("--aug_sigma_max", type=float, default=None)
    parser.add_argument("--aug_atom_frac", type=float, default=0.25)
    parser.add_argument("--aug_multiplier", type=int, default=1)

    # Misc
    parser.add_argument("--run_dir", type=str,
                        default="runs/ts1x_equiformerv2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_every_steps", type=int, default=6750)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Resume training from a saved best_model_*.pt "
                             "(restores model + optimizer + global_step + LR "
                             "schedule position). Default None = fresh start "
                             "(unchanged legacy behavior). NOTE: approximate "
                             "resume — DataLoader/aug RNG state is not restored.")

    args = parser.parse_args()

    # --- Resolve flags ---
    if args.aug_only:
        args.augment = True
    if args.no_linref:
        args.linref = False
    if args.aug_sigma is not None:
        args.aug_sigma_min = args.aug_sigma
        args.aug_sigma_max = args.aug_sigma
    if args.aug_sigma_max is None:
        args.aug_sigma_max = args.aug_sigma_min
    if args.init_mode == "rgd1" and args.rgd1_ckpt_path is None:
        parser.error("--init_mode=rgd1 requires --rgd1_ckpt_path")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Init mode: {args.init_mode} | Linref: {args.linref} | "
          f"Best metric: {args.best_metric}")

    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%m%d%H%M")

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, f"args_{timestamp}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Data ----
    print("Loading train dataset...")
    train_dataset = Ts1xDataset(args.train_lmdb)
    print(f"  Train: {len(train_dataset)}")
    print("Loading val dataset...")
    val_dataset = Ts1xDataset(args.val_lmdb)
    print(f"  Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- Linear reference ----
    linref = None
    linref_tensor = None
    if args.linref:
        if args.linref_path is not None:
            print(f"Loading pre-computed linref from {args.linref_path}")
            linref = np.load(args.linref_path, allow_pickle=True)["coeff"]
            z_names = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
            print("  Linear reference coefficients (eV):")
            for z in range(len(linref)):
                if linref[z] != 0:
                    print(f"    {z_names.get(z, f'Z={z}')} (Z={z}): {linref[z]:.4f}")
        else:
            print("Computing linear reference from train set...")
            linref = compute_linref_coeffs(
                train_dataset,
                max_z=args.linref_max_z,
                max_samples=args.linref_max_samples,
            )
            np.savez_compressed(
                os.path.join(args.run_dir, f"linref_{timestamp}.npz"),
                coeff=linref,
            )
        linref_tensor = torch.tensor(linref, dtype=torch.float32, device=device)

    # ---- Normalization ----
    energy_type = "residual" if args.linref else "raw"
    if args.norm_stats_path is not None:
        print(f"Loading pre-computed norm stats from {args.norm_stats_path}")
        with open(args.norm_stats_path) as f:
            norm_stats = json.load(f)
        # Keep only the three fields the training loop reads — drop any extras
        # (e.g., a stored linref list) to avoid confusion.
        norm_stats = {
            "energy_mean_per_atom": float(norm_stats["energy_mean_per_atom"]),
            "energy_std_per_atom": float(norm_stats["energy_std_per_atom"]),
            "force_std": float(norm_stats["force_std"]),
        }
    else:
        n_norm = (len(train_dataset) if args.norm_stats_max_samples is None
                  else min(len(train_dataset), args.norm_stats_max_samples))
        print(f"Computing normalization stats ({energy_type} energy) on "
              f"{n_norm:,} samples"
              f"{' (full train set)' if args.norm_stats_max_samples is None else ''}...")
        norm_stats = compute_normalization_stats(
            train_dataset, linref=linref,
            max_samples=args.norm_stats_max_samples,
        )
    if linref is not None:
        norm_stats["linref"] = linref.tolist()
    print(f"  Energy ({energy_type}): mean={norm_stats['energy_mean_per_atom']:.4f}, "
          f"std={norm_stats['energy_std_per_atom']:.4f} eV/atom")
    print(f"  Force std: {norm_stats['force_std']:.4f} eV/Å")
    with open(os.path.join(args.run_dir, f"norm_stats_{timestamp}.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    # ---- Model ----
    arch_key = args.init_mode if args.init_mode in MODEL_CONFIGS else "scratch"
    # rgd1 uses 10-element arch (RGD1 runs were max_num_elements=10); if future
    # RGD1 ckpts use a different arch, add a new config entry.
    if args.init_mode == "rgd1":
        arch_key = "scratch"
    cfg = MODEL_CONFIGS[arch_key]
    print(f"Building model ({arch_key} arch: N={cfg['num_layers']}, "
          f"L={cfg['lmax_list'][0]}, M={cfg['mmax_list'][0]})...")
    model = build_model(cfg, args.max_radius, args.max_neighbors)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if args.init_mode not in ("scratch", "horm_cfg", "horm_cfg_3l"):
        if args.init_mode == "oc20_31m":
            ckpt_path = args.oc20_31m_ckpt
        elif args.init_mode == "oc20_83m":
            ckpt_path = args.oc20_83m_ckpt
        else:
            ckpt_path = args.rgd1_ckpt_path
        print(f"Loading pretrained weights from {ckpt_path}")
        missing, unexpected = load_pretrained(model, ckpt_path, args.init_mode)
        print(f"  Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"    First missing: {missing[:5]}")
        if len(unexpected) > 0:
            print(f"    First unexpected: {unexpected[:5]}")

    model = model.to(device)

    # ---- Optimizer (head vs backbone LR split) ----
    head_modules = [model.energy_block, model.force_block]
    head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_param_ids]
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    head_lr = args.lr * args.head_lr_multiplier

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.max_steps, args.lr_min_factor,
    )

    # ---- Config header ----
    steps_per_epoch = len(train_loader)
    if args.aug_only:
        eff_batch = args.batch_size * args.aug_multiplier
        aug_mode_str = "aug_only"
    elif args.augment:
        eff_batch = args.batch_size * (1 + args.aug_multiplier)
        aug_mode_str = "orig+aug"
    else:
        eff_batch = args.batch_size
        aug_mode_str = "no_aug"

    if args.init_mode == "oc20_31m":
        ckpt_path_for_log = args.oc20_31m_ckpt
    elif args.init_mode == "oc20_83m":
        ckpt_path_for_log = args.oc20_83m_ckpt
    elif args.init_mode == "rgd1":
        ckpt_path_for_log = args.rgd1_ckpt_path
    elif args.init_mode == "horm_cfg":
        ckpt_path_for_log = "(none — scratch with HORM-paper arch)"
    elif args.init_mode == "horm_cfg_3l":
        ckpt_path_for_log = "(none — scratch with HORM-paper arch, 3 layers)"
    else:
        ckpt_path_for_log = "(none — scratch)"

    # Linref coefficient string (compact, inline in the header so the CSV alone is reproducible).
    if linref is not None:
        z_names = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
        linref_str = " ".join(
            f"{z_names.get(z, f'Z{z}')}:{linref[z]:.6f}"
            for z in range(len(linref)) if linref[z] != 0
        )
    else:
        linref_str = "(none)"

    # Environment info
    gpu_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu")
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "n/a"

    config_header = (
        f"# EquiformerV2 fine-tuning on Transition-1x (JMP v4 style)\n"
        f"# script: {os.path.basename(__file__)}\n"
        f"# timestamp(KST): {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# run_dir: {args.run_dir}\n"
        f"# sidecars: args_{timestamp}.json, norm_stats_{timestamp}.json"
        + (f", linref_{timestamp}.npz" if linref is not None else "") + "\n"
        f"# env: torch={torch_ver} cuda={cuda_ver} gpu={gpu_name}\n"
        f"# seed: {args.seed} num_workers: {args.num_workers}\n"
        f"# -- DATA --\n"
        f"# train_lmdb: {args.train_lmdb}\n"
        f"# val_lmdb:   {args.val_lmdb}\n"
        f"# train_samples: {len(train_dataset)}, val_samples: {len(val_dataset)}\n"
        f"# steps_per_epoch: {steps_per_epoch}\n"
        f"# -- MODEL --\n"
        f"# init_mode: {args.init_mode} | arch_key: {arch_key} | params: {n_params:,}\n"
        f"# ckpt_path: {ckpt_path_for_log}\n"
        f"# arch: num_layers={cfg['num_layers']} sphere_channels={cfg['sphere_channels']} "
        f"lmax={cfg['lmax_list'][0]} mmax={cfg['mmax_list'][0]} "
        f"grid_resolution={cfg['grid_resolution']} max_num_elements={cfg['max_num_elements']}\n"
        f"# arch: attn_hidden_channels={cfg['attn_hidden_channels']} num_heads={cfg['num_heads']} "
        f"attn_alpha_channels={cfg['attn_alpha_channels']} attn_value_channels={cfg['attn_value_channels']} "
        f"ffn_hidden_channels={cfg['ffn_hidden_channels']}\n"
        f"# arch: edge_channels={cfg['edge_channels']} num_distance_basis={cfg['num_distance_basis']} "
        f"norm_type=layer_norm_sh distance_function=gaussian\n"
        f"# arch: alpha_drop={cfg['alpha_drop']} drop_path_rate={cfg['drop_path_rate']} "
        f"proj_drop={cfg['proj_drop']} weight_init=uniform\n"
        f"# arch: attn_activation=silu ffn_activation=silu use_gate_act=False use_grid_mlp=True "
        f"use_sep_s2_act=True use_attn_renorm=True use_s2_act_attn=False\n"
        f"# max_radius={args.max_radius} max_neighbors={args.max_neighbors} "
        f"use_pbc=False regress_forces=True otf_graph=True\n"
        f"# -- LOSS / TRAIN --\n"
        f"# train_mode=direct-f eval_mode=direct-f+autograd best_metric={args.best_metric}\n"
        f"# loss: e=L1(per-atom normalized), f=pairwise_distance(p=2).mean() "
        f"(same as JMP/HORM v4)\n"
        f"# energy_coeff={args.energy_coeff} force_coeff={args.force_coeff} "
        f"clip_grad_norm={args.clip_grad_norm}\n"
        f"# epochs={args.epochs} batch_size={args.batch_size} "
        f"eval_batch_size={args.eval_batch_size}\n"
        f"# effective_batch_size={eff_batch} ({aug_mode_str})\n"
        f"# -- OPTIMIZER / SCHEDULER --\n"
        f"# optimizer=AdamW lr={args.lr} head_lr={head_lr:.6e} "
        f"(x{args.head_lr_multiplier}) weight_decay={args.weight_decay}\n"
        f"# scheduler=LambdaLR-cosine warmup_steps={args.warmup_steps} "
        f"max_steps={args.max_steps} lr_min_factor={args.lr_min_factor}\n"
        f"# -- AUGMENTATION --\n"
        f"# augment={args.augment} aug_only={args.aug_only} aug_method={args.aug_method} "
        f"aug_sigma=[{args.aug_sigma_min},{args.aug_sigma_max}] "
        f"aug_atom_frac={args.aug_atom_frac} aug_multiplier={args.aug_multiplier}\n"
        f"# -- NORMALIZATION --\n"
        f"# linref={args.linref} | coeffs(eV): {linref_str}\n"
        f"# energy_mean_per_atom={norm_stats['energy_mean_per_atom']:.6f} "
        f"energy_std_per_atom={norm_stats['energy_std_per_atom']:.6f} "
        f"force_std={norm_stats['force_std']:.6f}\n"
        f"# -- LOGGING --\n"
        f"# eval_every_steps={args.eval_every_steps} log_every={args.log_every} "
        f"save_best={args.save_best}\n"
        f"# ALL LOGGED METRICS ARE IN PHYSICAL UNITS: energy=eV/atom, force=eV/Å\n"
    )
    print(config_header)

    # ---- Log files ----
    best_val_f_mae = float("inf")
    global_step = 0
    start_epoch = 1

    # ---- Resume (optional; backward-compatible: default None = fresh start) ----
    if args.resume_ckpt is not None:
        print(f"Resuming from {args.resume_ckpt}")
        rck = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"], strict=True)
        if "optimizer_state_dict" in rck:
            optimizer.load_state_dict(rck["optimizer_state_dict"])
            print("  optimizer state restored")
        else:
            print("  WARNING: ckpt has no optimizer_state_dict — optimizer fresh")
        global_step = int(rck.get("global_step", 0))
        best_val_f_mae = float(
            rck.get("val_f_mae_autograd", float("inf"))
            if args.best_metric == "autograd"
            else rck.get("val_f_mae_direct", float("inf")))
        # LambdaLR is step-based: fast-forward to global_step so the cosine
        # schedule continues from where it left off (not restart from warmup).
        for _ in range(global_step):
            scheduler.step()
        start_epoch = global_step // max(1, steps_per_epoch) + 1
        print(f"  resumed: global_step={global_step} start_epoch={start_epoch} "
              f"best_val_f_mae({args.best_metric})={best_val_f_mae:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

    epoch_log_path = os.path.join(args.run_dir, f"epoch_log_{timestamp}.csv")
    epoch_log_file = open(epoch_log_path, "w")
    epoch_log_file.write(config_header)
    epoch_log_file.write("epoch,train_loss,train_e_mae_eV,train_f_mae_eV,lr,time\n")

    step_val_log_path = os.path.join(args.run_dir, f"step_val_log_{timestamp}.csv")
    step_val_log_file = open(step_val_log_path, "w")
    step_val_log_file.write(config_header)
    step_val_log_file.write(
        "global_step,epoch,val_e_mae,val_f_mae_direct,val_f_mae_autograd,lr\n")

    batch_log_path = os.path.join(args.run_dir, f"batch_log_{timestamp}.csv")
    batch_log_file = open(batch_log_path, "w")
    batch_log_file.write(config_header)
    batch_log_file.write("global_step,epoch,batch,loss,e_mae_eV,f_mae_eV,lr\n")

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} | LR={lr:.2e} | Step {global_step}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args, norm_stats,
            linref_tensor, epoch,
            dataset=train_dataset if args.augment else None,
            batch_log_file=batch_log_file,
            global_step=global_step,
            val_loader=val_loader,
            best_val_f_mae=best_val_f_mae,
            timestamp=timestamp,
            step_val_log_file=step_val_log_file,
        )

        global_step = train_metrics["global_step"]
        best_val_f_mae = train_metrics["best_val_f_mae"]

        elapsed = time.time() - t0
        print(f"  Epoch {epoch} done | Loss={train_metrics['loss']:.4f} "
              f"E_MAE={train_metrics['e_mae']:.4f} eV/atom "
              f"F_MAE={train_metrics['f_mae']:.4f} eV/Å | {elapsed:.1f}s")

        epoch_log_file.write(
            f"{epoch},{train_metrics['loss']:.6f},"
            f"{train_metrics['e_mae']:.6f},{train_metrics['f_mae']:.6f},"
            f"{lr:.2e},{elapsed:.1f}\n"
        )
        epoch_log_file.flush()

        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping.")
            break

    epoch_log_file.close()
    step_val_log_file.close()
    batch_log_file.close()

    torch.save({
        "global_step": global_step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "norm_stats": norm_stats,
        "linref": (linref_tensor.cpu().numpy().tolist()
                   if linref_tensor is not None else None),
        "args": vars(args),
    }, os.path.join(args.run_dir, f"final_model_{timestamp}.pt"))

    print(f"\nDone. Best val F_MAE ({args.best_metric}): {best_val_f_mae:.4f} eV/Å")
    print(f"Results: {args.run_dir}")


if __name__ == "__main__":
    main()