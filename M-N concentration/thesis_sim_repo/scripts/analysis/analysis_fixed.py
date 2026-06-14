#!/usr/bin/env python3
"""
analysis_fixed.py
=================
Core single-simulation analysis engine.

Key features:
  - Automatically repairs missing 'mol' column in dump files by loading
    molecule IDs from system.data (fixes the silent-zero orientation bug).
  - Reads the last N frames of a trajectory for production-run averaging.
  - Computes: number-density profiles, GDS location, water/methanol/NaCl
    orientation order parameters.
  - Saves PDF + PNG plots and a profiles.csv summary.

Usage (run inside a simulation directory):
  python3 analysis_fixed.py [--dump traj.lammpstrj] [--nframes 50]
                            [--binwidth 1.0] [--outdir fixed_results]
                            [--datafile system.data]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

# ── Atom-type constants ───────────────────────────────────────────────────────
T_O_WAT = 1; T_H_WAT = 2
T_C_MET = 3; T_O_MET = 4; T_H_MET = 5; T_H_OH = 6
T_NA    = 7; T_CL    = 8
METHANOL_TYPES = {T_C_MET, T_O_MET, T_H_MET, T_H_OH}
ION_TYPES      = {T_NA, T_CL}

BOX_LX, BOX_LY, BOX_Z = 60.0, 60.0, 200.0
DEFAULT_BINWIDTH  = 1.0
DEFAULT_NFRAMES   = 50
DEFAULT_DUMP      = "traj.lammpstrj"


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_mol_map_from_data(filename="system.data"):
    """
    Parse molecule IDs from a LAMMPS data file Atoms section.
    Returns {atom_id: mol_id} dict.
    Used to repair trajectories that lack a 'mol' column.
    """
    mol_map  = {}
    in_atoms = False
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Atoms") and not in_atoms:
                in_atoms = True
                continue
            if in_atoms:
                if line.startswith(("Bonds", "Angles", "Dihedrals", "Impropers")):
                    break
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mol_map[int(parts[0])] = int(parts[1])
                    except (ValueError, IndexError):
                        pass
    if not mol_map:
        raise RuntimeError(f"Could not parse molecule IDs from {filename}")
    return mol_map


def read_frames(filename, n_last, mol_map=None):
    """
    Read the last `n_last` frames from a LAMMPS dump file.
    If mol column is absent and mol_map is provided, inject mol IDs from map.
    Returns list of (natoms × 8) float64 arrays with columns:
      [atom_id, type, mol_id, _, x, y, z, _]
    """
    # First pass: collect byte offsets of each TIMESTEP header
    offsets = []
    with open(filename, "rb") as fh:
        while True:
            off  = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line.startswith(b"ITEM: TIMESTEP"):
                offsets.append(off)

    total = len(offsets)
    print(f"  Total frames in file: {total}")
    use_offsets = offsets[max(0, total - n_last):]
    print(f"  Reading last {len(use_offsets)} frames ...")

    frames = []
    with open(filename, "r") as fh:
        for off in use_offsets:
            fh.seek(off)
            fh.readline()          # ITEM: TIMESTEP
            fh.readline()          # timestep value
            fh.readline()          # ITEM: NUMBER OF ATOMS
            natoms = int(fh.readline().strip())
            fh.readline()          # ITEM: BOX BOUNDS
            fh.readline(); fh.readline(); fh.readline()   # box lines
            header = fh.readline().strip()
            cols   = header.split()[2:]                   # strip "ITEM: ATOMS"
            cmap   = {c: i for i, c in enumerate(cols)}

            idx_id   = cmap.get("id",   0)
            idx_type = cmap.get("type", 1)
            idx_mol  = cmap.get("mol", -1)
            idx_x    = cmap.get("x",   4)
            idx_y    = cmap.get("y",   5)
            idx_z    = cmap.get("z",   6)
            has_mol  = (idx_mol >= 0)

            data = np.empty((natoms, 8), dtype=np.float64)
            for i in range(natoms):
                parts     = fh.readline().split()
                atom_id   = int(parts[idx_id])
                data[i, 0] = atom_id
                data[i, 1] = float(parts[idx_type])
                data[i, 2] = float(parts[idx_mol]) if has_mol else (
                    mol_map.get(atom_id, 0) if mol_map else 0.0)
                data[i, 4] = float(parts[idx_x])
                data[i, 5] = float(parts[idx_y])
                data[i, 6] = float(parts[idx_z])
            frames.append(data)

    print(f"  Loaded {len(frames)} frames successfully")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def build_mol_type_map(frame):
    """Return {mol_id: set_of_atom_types} for one frame."""
    m = {}
    for mid, at in zip(frame[:, 2].astype(int), frame[:, 1].astype(int)):
        m.setdefault(mid, set()).add(at)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Density profiles
# ─────────────────────────────────────────────────────────────────────────────

def density_profile(frames, type_set, binwidth, lx, ly, lz):
    """
    Time-averaged 1D number density profile along z for atoms in `type_set`.
    Returns (z_centres, density_in_atoms_per_Å³).
    """
    nbins  = int(lz / binwidth)
    edges  = np.linspace(0, lz, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol    = binwidth * lx * ly
    acc    = np.zeros(nbins)
    for f in frames:
        mask = np.isin(f[:, 1].astype(int), list(type_set))
        z    = f[mask, 6]
        hist, _ = np.histogram(z, bins=edges)
        acc += hist / vol
    return centers, acc / len(frames)


def find_gds(centers, water_density, lz):
    """
    Locate the Gibbs Dividing Surface (GDS) positions at 50% bulk water density.
    Returns (z_bottom_GDS, z_top_GDS) in Å.
    """
    bulk       = (centers > 0.3 * lz) & (centers < 0.7 * lz)
    rho_bulk   = np.mean(water_density[bulk]) if bulk.any() else water_density.max()
    thresh     = 0.5 * rho_bulk
    half       = len(centers) // 2

    z_bot = centers[0]
    for i in range(half):
        if water_density[i] >= thresh:
            z_bot = centers[i]
            break

    z_top = centers[-1]
    for i in range(len(centers) - 1, half, -1):
        if water_density[i] >= thresh:
            z_top = centers[i]
            break

    return z_bot, z_top


# ─────────────────────────────────────────────────────────────────────────────
# Orientation order parameters
# ─────────────────────────────────────────────────────────────────────────────

def orientation_water(frames, binwidth, lz):
    """
    Water dipole orientation ⟨cos θ⟩ profile.
    θ = angle between the dipole vector (O→midpoint-of-H's) and the +z axis.
    """
    nbins   = int(lz / binwidth)
    edges   = np.linspace(0, lz, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    cos_acc = np.zeros(nbins)
    cnt_acc = np.zeros(nbins)

    for f in frames:
        mol_ids        = f[:, 2].astype(int)
        atypes         = f[:, 1].astype(int)
        xyz            = f[:, 4:7]
        mol_type_map   = build_mol_type_map(f)

        for mid, types in mol_type_map.items():
            if types != {T_O_WAT, T_H_WAT}:
                continue
            idx   = np.where(mol_ids == mid)[0]
            if len(idx) != 3:
                continue
            o_idx = idx[atypes[idx] == T_O_WAT]
            h_idx = idx[atypes[idx] == T_H_WAT]
            if len(o_idx) != 1 or len(h_idx) != 2:
                continue
            o_pos  = xyz[o_idx[0]]
            h_mid  = (xyz[h_idx[0]] + xyz[h_idx[1]]) / 2
            dipole = h_mid - o_pos
            norm   = np.linalg.norm(dipole)
            if norm < 1e-12:
                continue
            cos_t = dipole[2] / norm
            bi    = int(o_pos[2] / binwidth)
            if 0 <= bi < nbins:
                cos_acc[bi] += cos_t
                cnt_acc[bi] += 1

    avg = np.divide(cos_acc, cnt_acc,
                    out=np.zeros_like(cos_acc), where=cnt_acc > 0)
    return centers, avg


def orientation_methanol(frames, binwidth, lz):
    """
    Methanol O–H orientation ⟨cos θ⟩ profile.
    θ = angle between O→H(hydroxyl) vector and the +z axis.
    """
    nbins   = int(lz / binwidth)
    edges   = np.linspace(0, lz, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    cos_acc = np.zeros(nbins)
    cnt_acc = np.zeros(nbins)

    for f in frames:
        mol_ids      = f[:, 2].astype(int)
        atypes       = f[:, 1].astype(int)
        xyz          = f[:, 4:7]
        mol_type_map = build_mol_type_map(f)

        for mid, types in mol_type_map.items():
            if not METHANOL_TYPES.issubset(types):
                continue
            if T_O_WAT in types or T_H_WAT in types:
                continue
            idx   = np.where(mol_ids == mid)[0]
            o_idx = idx[atypes[idx] == T_O_MET]
            h_idx = idx[atypes[idx] == T_H_OH]
            if len(o_idx) != 1 or len(h_idx) != 1:
                continue
            oh_vec = xyz[h_idx[0]] - xyz[o_idx[0]]
            norm   = np.linalg.norm(oh_vec)
            if norm < 1e-12:
                continue
            cos_t = oh_vec[2] / norm
            bi    = int(xyz[o_idx[0], 2] / binwidth)
            if 0 <= bi < nbins:
                cos_acc[bi] += cos_t
                cnt_acc[bi] += 1

    avg = np.divide(cos_acc, cnt_acc,
                    out=np.zeros_like(cos_acc), where=cnt_acc > 0)
    return centers, avg


def orientation_nacl_zonal(frames, binwidth, lz, z_bot, z_top):
    """
    NaCl pair orientation ⟨cos θ⟩ profile.
    For each Na⁺, finds the nearest Cl⁻ and records the z-component of
    the Na→Cl unit vector. Bins with < 3 samples are set to NaN.
    """
    MIN_SAMPLES = 3
    nbins   = int(lz / binwidth)
    edges   = np.linspace(0, lz, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    cos_acc = np.zeros(nbins)
    cnt_acc = np.zeros(nbins)

    for f in frames:
        atypes = f[:, 1].astype(int)
        xyz    = f[:, 4:7]
        na_pos = xyz[atypes == T_NA]
        cl_pos = xyz[atypes == T_CL]
        if len(na_pos) == 0 or len(cl_pos) == 0:
            continue
        for na in na_pos:
            dist    = np.linalg.norm(cl_pos - na, axis=1)
            nearest = cl_pos[np.argmin(dist)]
            vec     = nearest - na
            norm    = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            bi = int(na[2] / binwidth)
            if 0 <= bi < nbins:
                cos_acc[bi] += vec[2] / norm
                cnt_acc[bi] += 1
        for cl in cl_pos:
            dist    = np.linalg.norm(na_pos - cl, axis=1)
            nearest = na_pos[np.argmin(dist)]
            vec     = nearest - cl
            norm    = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            bi = int(cl[2] / binwidth)
            if 0 <= bi < nbins:
                cos_acc[bi] += vec[2] / norm
                cnt_acc[bi] += 1

    avg             = np.full(nbins, np.nan)
    mask            = cnt_acc >= MIN_SAMPLES
    avg[mask]       = cos_acc[mask] / cnt_acc[mask]
    return centers, avg


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

PLOT_STYLE = {
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
}


def save_line_plot(x, y, title, xlabel, ylabel, filepath,
                   color="steelblue", ylim=None, zero_line=False,
                   gds=None, nan_ok=False):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        if nan_ok:
            mask = ~np.isnan(y)
            ax.plot(x[mask], y[mask], "o", color=color, ms=3, alpha=0.7)
        else:
            ax.plot(x, y, color=color, lw=1.5)
        if zero_line:
            ax.axhline(0, color="black", lw=0.8, ls="--")
        if gds:
            ax.axvline(gds[0], color="gray", lw=1, ls=":",
                       label=f"GDS bottom ({gds[0]:.1f} Å)")
            ax.axvline(gds[1], color="gray", lw=1, ls=":",
                       label=f"GDS top ({gds[1]:.1f} Å)")
            ax.legend(fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlim(0, BOX_Z)
        if ylim:
            ax.set_ylim(ylim)
        plt.tight_layout()
        fig.savefig(filepath + ".pdf", bbox_inches="tight", dpi=300)
        fig.savefig(filepath + ".png", bbox_inches="tight", dpi=300)
        plt.close(fig)


def save_na_cl_plot(x, na, cl, title, filepath, gds=None):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, na, color="#E67E22", lw=1.5, label="Na⁺")
        ax.plot(x, cl, color="#27AE60", lw=1.5, label="Cl⁻")
        if gds:
            ax.axvline(gds[0], color="gray", lw=1, ls=":")
            ax.axvline(gds[1], color="gray", lw=1, ls=":")
        ax.set_xlabel("Z (Å)")
        ax.set_ylabel("Number Density (atoms/Å³)")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlim(0, BOX_Z)
        ax.legend()
        plt.tight_layout()
        fig.savefig(filepath + ".pdf", bbox_inches="tight", dpi=300)
        fig.savefig(filepath + ".png", bbox_inches="tight", dpi=300)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-simulation analysis with mol-ID repair")
    parser.add_argument("--dump",     default=DEFAULT_DUMP)
    parser.add_argument("--nframes",  default=DEFAULT_NFRAMES, type=int)
    parser.add_argument("--binwidth", default=DEFAULT_BINWIDTH, type=float)
    parser.add_argument("--outdir",   default="fixed_results")
    parser.add_argument("--datafile", default="system.data")
    args = parser.parse_args()

    bw = args.binwidth
    lz = BOX_Z

    # Locate dump file
    if not os.path.exists(args.dump):
        for alt in ["traj_combined.lammpstrj",
                    "traj_new.lammpstrj",
                    "traj_continued.lammpstrj"]:
            if os.path.exists(alt):
                args.dump = alt
                break
    if not os.path.exists(args.dump):
        sys.exit(f"ERROR: trajectory file '{args.dump}' not found")

    # Load mol map
    mol_map = None
    if os.path.exists(args.datafile):
        print(f"  Loading mol mapping from {args.datafile} ...")
        mol_map = load_mol_map_from_data(args.datafile)
    else:
        print("  WARNING: system.data not found – mol IDs unavailable")

    sim_name = os.path.basename(os.getcwd())
    out_dir  = os.path.join(args.outdir, sim_name)
    os.makedirs(out_dir, exist_ok=True)

    frames = read_frames(args.dump, args.nframes, mol_map=mol_map)

    print("  Computing density profiles ...")
    z, rho_water = density_profile(frames, {T_O_WAT},    bw, BOX_LX, BOX_LY, lz)
    _, rho_meoh  = density_profile(frames, METHANOL_TYPES, bw, BOX_LX, BOX_LY, lz)
    _, rho_nacl  = density_profile(frames, ION_TYPES,    bw, BOX_LX, BOX_LY, lz)
    _, rho_na    = density_profile(frames, {T_NA},       bw, BOX_LX, BOX_LY, lz)
    _, rho_cl    = density_profile(frames, {T_CL},       bw, BOX_LX, BOX_LY, lz)

    z_bot, z_top = find_gds(z, rho_water, lz)
    gds = (z_bot, z_top)
    print(f"  GDS: bottom = {z_bot:.1f} Å,  top = {z_top:.1f} Å")

    print("  Computing orientation order parameters ...")
    _, w_orient = orientation_water(frames, bw, lz)
    _, m_orient = orientation_methanol(frames, bw, lz)
    _, n_orient = orientation_nacl_zonal(frames, bw, lz, z_bot, z_top)

    # ── Save plots ────────────────────────────────────────────────────────────
    p = lambda n: os.path.join(out_dir, n)

    save_line_plot(z, rho_meoh, f"Methanol Density – {sim_name}",
                   "Z (Å)", "atoms/Å³", p("01_methanol_density"),
                   color="#2196F3", gds=gds)
    save_line_plot(z, rho_nacl, f"NaCl Density – {sim_name}",
                   "Z (Å)", "atoms/Å³", p("02_nacl_density"),
                   color="#9C27B0", gds=gds)
    save_na_cl_plot(z, rho_na, rho_cl,
                    f"Na⁺ vs Cl⁻ Density – {sim_name}", p("03_na_vs_cl"), gds=gds)
    save_line_plot(z, w_orient, f"Water Dipole Orientation – {sim_name}",
                   "Z (Å)", r"$\langle\cos\theta\rangle$",
                   p("04_water_orientation"), color="#2196F3",
                   ylim=(-1, 1), zero_line=True, gds=gds)
    save_line_plot(z, m_orient, f"Methanol O–H Orientation – {sim_name}",
                   "Z (Å)", r"$\langle\cos\theta\rangle$",
                   p("05_methanol_orientation"), color="#F44336",
                   ylim=(-1, 1), zero_line=True, gds=gds)
    save_line_plot(z, n_orient, f"NaCl Pair Orientation – {sim_name}",
                   "Z (Å)", r"$\langle\cos\theta\rangle$",
                   p("06_nacl_orientation"), color="#4CAF50",
                   ylim=(-1, 1), zero_line=True, gds=gds, nan_ok=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    n_fill = np.where(np.isnan(n_orient), 0.0, n_orient)
    np.savetxt(
        os.path.join(out_dir, "profiles.csv"),
        np.column_stack([z, rho_water, rho_meoh, rho_nacl,
                         rho_na, rho_cl, w_orient, m_orient, n_fill]),
        delimiter=",",
        header="z_A,rho_water,rho_methanol,rho_nacl,rho_na,rho_cl,"
               "w_orient,m_orient,n_orient",
        comments="",
    )

    print(f"  Done. Results saved in: {out_dir}/")


if __name__ == "__main__":
    main()
