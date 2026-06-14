#!/usr/bin/env python3
"""
hbond_profile.py
================
Vectorised, PBC-corrected hydrogen bond profile along the z-axis.

Criteria (geometric, standard):
  r(O–O)  < 3.50 Å
  r(H–O_acceptor) < 3.50 Å
  ∠(O_donor–H–O_acceptor) > 150°

Requires scipy for cKDTree (fast neighbour search).
Install: pip install scipy --break-system-packages

Output:
  hbond_profile_comparison.pdf / .png  – ⟨H-bonds per water molecule⟩ vs z

Usage (from repo root):
  # Single system:
  cd simulations/sim_M10_N10
  python3 ../../scripts/analysis/hbond_profile.py --dump traj.lammpstrj

  # All systems (run_all mode):
  python3 scripts/analysis/hbond_profile.py --run_all
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, os, sys

try:
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("scipy not found. Install with:  pip install scipy --break-system-packages")

BOX_LX, BOX_LY, BOX_Z = 60.0, 60.0, 200.0
BINWIDTH    = 1.0   # Å
T_O_WAT     = 1
T_H_WAT     = 2
OO_CUTOFF   = 3.50  # Å
HOO_CUTOFF  = 3.50  # Å
ANGLE_MIN   = 150.0 # degrees

SIM_LABELS = ["sim_N10", "sim_M10",
              "sim_M10_N10", "sim_M20_N20", "sim_M50_N50", "sim_M100_N100"]
LABEL_MAP  = {
    "sim_N10":      "N10",
    "sim_M10":      "M10",
    "sim_M10_N10":  "M10N10",
    "sim_M20_N20":  "M20N20",
    "sim_M50_N50":  "M50N50",
    "sim_M100_N100":"M100N100",
}
SYSTEM_COLORS = {
    "N10":     "#888780",
    "M10":     "#1D9E75",
    "M10N10":  "#378ADD",
    "M20N20":  "#EF9F27",
    "M50N50":  "#D85A30",
    "M100N100":"#7F77DD",
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_frames_simple(dumpfile, max_frames):
    """Lightweight frame reader that handles 6- or 7-column atom lines."""
    frames = []
    with open(dumpfile) as fh:
        raw = fh.readlines()
    i = 0; n = len(raw)
    while i < n:
        if "ITEM: TIMESTEP" not in raw[i]:
            i += 1; continue
        i += 2
        n_atoms  = int(raw[i].strip()); i += 1
        i += 4                           # skip BOX BOUNDS
        atom_lines = raw[i:i + n_atoms]; i += n_atoms
        data = np.array([l.split() for l in atom_lines], dtype=float)
        nc   = data.shape[1]
        frame = {}
        if nc >= 7:
            frame["id"]   = data[:, 0].astype(int)
            frame["mol"]  = data[:, 1].astype(int)
            frame["type"] = data[:, 2].astype(int)
            frame["xyz"]  = data[:, 4:7]
        elif nc == 6:
            frame["id"]   = data[:, 0].astype(int)
            frame["mol"]  = data[:, 1].astype(int)
            frame["type"] = data[:, 2].astype(int)
            frame["xyz"]  = data[:, 3:6]
        else:
            frame["id"]   = data[:, 0].astype(int)
            frame["mol"]  = np.zeros(n_atoms, int)
            frame["type"] = data[:, 1].astype(int)
            frame["xyz"]  = data[:, 2:5]
        frames.append(frame)
    if len(frames) > max_frames:
        frames = frames[-max_frames:]
    return frames


def load_mol_map(datafile):
    mol_map  = {}; in_atoms = False
    with open(datafile) as fh:
        for line in fh:
            ls = line.strip()
            if ls.startswith("Atoms"):
                in_atoms = True; continue
            if in_atoms:
                if not ls or ls.startswith("#"):
                    continue
                if any(ls.startswith(k) for k in
                       ["Bonds","Angles","Dihedrals","Impropers","Velocities","Masses"]):
                    break
                parts = ls.split()
                if len(parts) >= 3:
                    mol_map[int(parts[0])] = int(parts[1])
    return mol_map


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

def mic_xy(dr, lx, ly):
    """Apply minimum image convention in x and y (not z for slab)."""
    dr = dr.copy()
    dr[:, 0] -= lx * np.round(dr[:, 0] / lx)
    dr[:, 1] -= ly * np.round(dr[:, 1] / ly)
    return dr


def build_mol_index_map(frame):
    """Map mol_id → {'O': atom_row_index, 'H': [row1, row2]} for water only."""
    mol_map = {}
    types = frame["type"]; mols = frame["mol"]
    for i in range(len(types)):
        t = types[i]
        if t not in (T_O_WAT, T_H_WAT):
            continue
        mid = mols[i]
        if mid not in mol_map:
            mol_map[mid] = {"O": None, "H": []}
        if t == T_O_WAT:
            mol_map[mid]["O"] = i
        else:
            mol_map[mid]["H"].append(i)
    # Keep only complete water molecules
    return {k: v for k, v in mol_map.items()
            if v["O"] is not None and len(v["H"]) == 2}


def hbond_count_profile(frames, binwidth=BINWIDTH):
    """
    Returns (z_centres, avg_hbonds_per_water_molecule) arrays.
    Uses vectorised cKDTree neighbour search with MIC correction.
    """
    nbins     = int(BOX_Z / binwidth)
    edges     = np.linspace(0, BOX_Z, nbins + 1)
    centers   = (edges[:-1] + edges[1:]) / 2
    hbond_sum = np.zeros(nbins)
    donor_cnt = np.zeros(nbins)

    # Build stable row-index arrays from first frame
    mol_index_map = build_mol_index_map(frames[0])
    mol_ids_list  = list(mol_index_map.keys())
    o_rows  = np.array([mol_index_map[m]["O"]    for m in mol_ids_list])
    h1_rows = np.array([mol_index_map[m]["H"][0] for m in mol_ids_list])
    h2_rows = np.array([mol_index_map[m]["H"][1] for m in mol_ids_list])

    for frame in frames:
        xyz    = frame["xyz"]
        o_pos  = xyz[o_rows]
        h1_pos = xyz[h1_rows]
        h2_pos = xyz[h2_rows]

        tree  = cKDTree(o_pos)
        pairs = tree.query_pairs(r=OO_CUTOFF + 2.0, output_type="ndarray")
        if pairs.shape[0] == 0:
            continue

        d_idx = pairs[:, 0]; a_idx = pairs[:, 1]
        oo_vec  = o_pos[a_idx] - o_pos[d_idx]
        oo_vec  = mic_xy(oo_vec, BOX_LX, BOX_LY)
        oo_dist = np.linalg.norm(oo_vec, axis=1)
        valid   = oo_dist < OO_CUTOFF
        d_idx   = d_idx[valid]; a_idx = a_idx[valid]

        # Both directions: d→a and a→d
        all_donors    = np.concatenate([d_idx, a_idx])
        all_acceptors = np.concatenate([a_idx, d_idx])

        for h_pos_all in [h1_pos, h2_pos]:
            hod_vec  = o_pos[all_donors]    - h_pos_all[all_donors]
            hod_vec  = mic_xy(hod_vec, BOX_LX, BOX_LY)
            hod_dist = np.linalg.norm(hod_vec, axis=1)

            hoa_vec  = o_pos[all_acceptors] - h_pos_all[all_donors]
            hoa_vec  = mic_xy(hoa_vec, BOX_LX, BOX_LY)
            hoa_dist = np.linalg.norm(hoa_vec, axis=1)

            close     = hoa_dist < HOO_CUTOFF
            cos_angle = np.full(len(all_donors), -2.0)
            if np.any(close):
                denom    = hod_dist[close] * hoa_dist[close]
                safe     = denom > 1e-8
                cos_v    = (np.einsum("ij,ij->i",
                                      hod_vec[close], hoa_vec[close])
                            / np.where(safe, denom, 1.0))
                cos_angle[close] = np.where(safe, cos_v, -2.0)

            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            bonded    = angle_deg > ANGLE_MIN
            donor_z   = o_pos[all_donors[bonded], 2] % BOX_Z
            bi        = np.clip(np.floor(donor_z / binwidth).astype(int),
                                0, nbins - 1)
            hbond_sum += np.bincount(bi, minlength=nbins).astype(float)

        donor_z = o_pos[:, 2] % BOX_Z
        bi      = np.clip(np.floor(donor_z / binwidth).astype(int),
                          0, nbins - 1)
        donor_cnt += np.bincount(bi, minlength=nbins).astype(float)

    avg_hbond = np.divide(hbond_sum, donor_cnt,
                          out=np.zeros_like(hbond_sum), where=donor_cnt > 0)
    return centers, avg_hbond


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_profiles(profiles_dict, outfile="hbond_profile_comparison.pdf"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (z, hb) in profiles_dict.items():
        ax.plot(z, hb, lw=1.8, label=label,
                color=SYSTEM_COLORS.get(label, None))
    ax.set_xlabel("z (Å)", fontsize=12)
    ax.set_ylabel("H-bonds per water molecule", fontsize=12)
    ax.set_xlim(0, BOX_Z); ax.set_ylim(bottom=0)
    ax.axhline(3.5, color="gray", lw=0.8, ls="--", label="Bulk SPC/Fw ref. (~3.5)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_single(dumpfile, datafile=None, nframes=50, label="system", outdir="."):
    mol_map = load_mol_map(datafile) if (datafile and os.path.exists(datafile)) else None
    frames  = read_frames_simple(dumpfile, nframes)
    if mol_map:
        for frame in frames:
            for i, aid in enumerate(frame["id"]):
                frame["mol"][i] = mol_map.get(int(aid), frame["mol"][i])
    z, hb = hbond_count_profile(frames)
    outfile = os.path.join(outdir, f"hbond_{label}.pdf")
    plot_profiles({label: (z, hb)}, outfile)
    return z, hb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump",    default="traj.lammpstrj")
    parser.add_argument("--data",    default="system.data")
    parser.add_argument("--nframes", default=50, type=int)
    parser.add_argument("--label",   default=None)
    parser.add_argument("--outdir",  default=".")
    parser.add_argument("--run_all", action="store_true",
                        help="Process all known simulation directories from repo root")
    args = parser.parse_args()

    if args.run_all:
        base     = os.getcwd()
        profiles = {}
        for subdir in SIM_LABELS:
            spath = os.path.join(base, "simulations", subdir)
            if not os.path.isdir(spath):
                continue
            dump = None
            for cand in ["traj.lammpstrj", "traj_combined.lammpstrj"]:
                p = os.path.join(spath, cand)
                if os.path.exists(p):
                    dump = p; break
            if dump is None:
                continue
            datafile = os.path.join(spath, "system.data")
            label    = LABEL_MAP.get(subdir, subdir)
            print(f"  Processing {label} ...")
            z, hb = run_single(dump, datafile, args.nframes,
                                label=label, outdir=base)
            profiles[label] = (z, hb)
        if profiles:
            plot_profiles(profiles, "hbond_profile_comparison.pdf")
    else:
        label = args.label or os.path.basename(os.getcwd())
        run_single(args.dump, args.data, args.nframes,
                   label=label, outdir=args.outdir)


if __name__ == "__main__":
    main()
