#!/usr/bin/env python3
"""
surface_excess.py
=================
Block-averaged surface excess Γ (methanol) using the Gibbs adsorption formalism.

Method:
  The trajectory is divided into N_BLOCKS blocks of BLOCK_NS ns each.
  For each block the excess is computed as:

      Γ = ∫ [ρ_methanol(z) − ρ_bulk_methanol] dz  (over interface regions)

  divided by 6 (atoms per methanol molecule) to give molecules/Å².
  The reported uncertainty is the standard deviation across blocks.

Output:
  surface_excess_blocks.txt  – per-block Γ values
  surface_excess_blocks.pdf/png – bar chart with error cap

Usage (run inside a simulation directory):
  python3 surface_excess.py [--dump traj.lammpstrj] [--data system.data]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from analysis_fixed import (
    BOX_LX, BOX_LY, BOX_Z,
    T_O_WAT, METHANOL_TYPES,
    load_mol_map_from_data, read_frames,
    density_profile, find_gds,
)

BINWIDTH      = 1.0    # Å
BLOCK_NS      = 1.0    # ns per block
TOTAL_NS      = 5.0    # production run length
FRAMES_PER_NS = 1000   # dump frequency: every 1 ps
N_BLOCKS      = int(TOTAL_NS / BLOCK_NS)


def compute_gamma(frames_subset):
    """Compute methanol surface excess (molecules/Å²) for a frame subset."""
    z, rho_water = density_profile(
        frames_subset, {T_O_WAT}, BINWIDTH, BOX_LX, BOX_LY, BOX_Z)

    bulk_mask    = (z > 0.3 * BOX_Z) & (z < 0.7 * BOX_Z)
    rho_bulk_w   = np.mean(rho_water[bulk_mask])
    thresh       = 0.5 * rho_bulk_w
    half         = len(z) // 2

    z_bot = z[0]
    for i in range(half):
        if rho_water[i] >= thresh:
            z_bot = z[i]; break
    z_top = z[-1]
    for i in range(len(z) - 1, half, -1):
        if rho_water[i] >= thresh:
            z_top = z[i]; break

    _, rho_meoh    = density_profile(
        frames_subset, METHANOL_TYPES, BINWIDTH, BOX_LX, BOX_LY, BOX_Z)

    bulk_meoh_mask = (z > z_bot + 5) & (z < z_top - 5)
    rho_bulk_meoh  = (np.mean(rho_meoh[bulk_meoh_mask])
                      if np.any(bulk_meoh_mask) else 0.0)

    # Integrate excess at both interfaces
    idx_bot     = np.searchsorted(z, z_bot)
    excess_bot  = np.trapz(rho_meoh[:idx_bot] - rho_bulk_meoh, z[:idx_bot])
    idx_top     = np.searchsorted(z, z_top)
    excess_top  = np.trapz(rho_meoh[idx_top:]  - rho_bulk_meoh, z[idx_top:])

    total_atoms = excess_bot + excess_top
    return total_atoms / 6.0   # atoms → molecules/Å²


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="traj.lammpstrj")
    parser.add_argument("--data", default="system.data")
    args = parser.parse_args()

    # Locate trajectory
    if not os.path.exists(args.dump):
        for alt in ["traj_combined.lammpstrj",
                    "traj_new.lammpstrj",
                    "traj_continued.lammpstrj"]:
            if os.path.exists(alt):
                args.dump = alt; break
    if not os.path.exists(args.dump):
        sys.exit(f"ERROR: trajectory '{args.dump}' not found")

    mol_map = (load_mol_map_from_data(args.data)
               if os.path.exists(args.data) else None)

    n_frames_to_read = N_BLOCKS * int(FRAMES_PER_NS * BLOCK_NS)
    print(f"  Reading up to {n_frames_to_read} frames for {N_BLOCKS} blocks ...")
    frames = read_frames(args.dump, n_frames_to_read, mol_map=mol_map)

    block_gamma = []
    block_size  = int(FRAMES_PER_NS * BLOCK_NS)

    for b in range(N_BLOCKS):
        start = b * block_size
        end   = start + block_size
        if end > len(frames):
            print(f"  WARNING: only {len(frames)} frames available; "
                  f"stopping at block {b}")
            break
        gamma = compute_gamma(frames[start:end])
        block_gamma.append(gamma)
        print(f"    Block {b+1}/{N_BLOCKS}:  Γ = {gamma:.4e} molecules/Å²")

    block_gamma = np.array(block_gamma)
    mean_gamma  = np.mean(block_gamma)
    std_gamma   = np.std(block_gamma, ddof=1)

    print(f"\n  Γ (methanol surface excess)")
    print(f"    Mean : {mean_gamma:.4e} molecules/Å²")
    print(f"    Std  : {std_gamma:.4e} molecules/Å²  (n={len(block_gamma)} blocks)")

    np.savetxt("surface_excess_blocks.txt", block_gamma,
               header="block_gamma_mol_per_A2")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(block_gamma) + 1),
           block_gamma * 1e3,
           yerr=std_gamma * 1e3,
           capsize=5, color="steelblue", alpha=0.8, edgecolor="black")
    ax.axhline(mean_gamma * 1e3, color="red", lw=1.5, ls="--",
               label=f"Mean = {mean_gamma*1e3:.3f} ×10⁻³")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Γ  (×10⁻³ molecules/Å²)")
    ax.set_title("Methanol Surface Excess – Block Average")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("surface_excess_blocks.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("surface_excess_blocks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: surface_excess_blocks.pdf / .png / .txt")


if __name__ == "__main__":
    main()
