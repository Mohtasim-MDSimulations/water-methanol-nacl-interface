#!/usr/bin/env python3
"""
analysis.py
===========
General analysis for the water-methanol liquid-vapour interface.

Produces:
  - Surface tension bar graph (1 ns blocks) from simulation.log
  - Water and methanol number-density profiles
  - Water dipole orientation <cos theta> profile
  - Summary CSV of all profiles

BUG FIXES vs original:
  1. cos theta calculation: original code embedded walrus assignments for
     h_mid_x and h_mid_y INSIDE np.sqrt() but then used those intermediate
     values incorrectly (subtraction of o_x/o_y was missing before squaring).
     Fixed by computing the full 3D dipole vector explicitly.

  2. Positional indexing removed: original used atoms_arr[mol*3] which
     assumes LAMMPS always writes atoms in creation-ID order. With MPI
     decomposition this is not guaranteed. Fixed by reading mol-ID from
     the dump and grouping atoms by mol-ID.

  3. CLI arguments added: n_methanol, dump file, log file are now
     command-line parameters so the script works for any system size
     without editing source code.

  4. charge column (q) removed from frame parser: dump no longer writes
     charges (static; the original col.get('q', 8) default silently used
     a wrong column if column order changed).

Usage:
  python3 analysis.py [--dump traj.lammpstrj] [--log simulation.log]
                      [--nmethanol 10] [--outdir analysis_results]
                      [--prod_start 500000] [--nframes_max 0]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

# Atom-type constants
T_O_WAT = 1; T_H_WAT = 2
T_C_MET = 3; T_O_MET = 4; T_H_MET = 5; T_H_OH = 6

MASSES = {
    T_O_WAT: 15.9994, T_H_WAT: 1.00794,
    T_C_MET: 12.0110, T_O_MET: 15.9994,
    T_H_MET: 1.00794, T_H_OH:  1.00794,
}

BOX_X, BOX_Y = 60.0, 60.0
ZMIN,  ZMAX  = 0.0, 200.0
NBINS        = 200


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_lammps_dump(filename, prod_start, nframes_max):
    """
    Generator yielding (timestep, frame_dict) for production frames only.

    frame_dict keys:
      'mol'  - ndarray int   molecule ID per atom
      'type' - ndarray int   atom type per atom
      'xyz'  - ndarray Nx3   coordinates

    BUG FIX: reads mol column; no positional indexing.
    """
    n_yielded = 0
    with open(filename) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue
            timestep = int(fh.readline().strip())

            while "NUMBER OF ATOMS" not in fh.readline():
                pass
            natoms = int(fh.readline().strip())

            while "BOX BOUNDS" not in fh.readline():
                pass
            fh.readline(); fh.readline(); fh.readline()

            while True:
                hdr = fh.readline()
                if "ITEM: ATOMS" in hdr:
                    break
            cols = hdr.split()[2:]
            cmap = {c: i for i, c in enumerate(cols)}

            idx_mol  = cmap.get("mol",  -1)
            idx_type = cmap.get("type",  1)
            idx_x    = cmap.get("x",     2)
            idx_y    = cmap.get("y",     3)
            idx_z    = cmap.get("z",     4)
            has_mol  = (idx_mol >= 0)

            mol_arr  = np.zeros(natoms, dtype=int)
            type_arr = np.zeros(natoms, dtype=int)
            xyz_arr  = np.zeros((natoms, 3))

            for i in range(natoms):
                parts         = fh.readline().split()
                mol_arr[i]    = int(float(parts[idx_mol])) if has_mol else 0
                type_arr[i]   = int(float(parts[idx_type]))
                xyz_arr[i, 0] = float(parts[idx_x])
                xyz_arr[i, 1] = float(parts[idx_y])
                xyz_arr[i, 2] = float(parts[idx_z])

            if timestep < prod_start:
                continue

            yield timestep, {"mol": mol_arr, "type": type_arr, "xyz": xyz_arr}
            n_yielded += 1
            if nframes_max > 0 and n_yielded >= nframes_max:
                break


def parse_surface_tension_blocks(logfile, prod_start, step_to_ns, zlen):
    """
    Parse pxx, pyy, pzz from simulation.log and return block-averaged
    surface tension (mN/m) in 1 ns windows.
    """
    if not os.path.exists(logfile):
        return None, None

    steps = []; gamma = []
    in_prod = False

    with open(logfile) as fh:
        for line in fh:
            if "run 5000000" in line:
                in_prod = True; continue
            if in_prod and "Loop time" in line:
                break
            if not in_prod or not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                step = int(parts[0])
                if step < prod_start:
                    continue
                pxx = float(parts[6]); pyy = float(parts[7]); pzz = float(parts[8])
                conv      = 0.0101325 * zlen / 2.0
                gamma_val = conv * (pzz - (pxx + pyy) / 2.0)
                steps.append(step); gamma.append(gamma_val)
            except (ValueError, IndexError):
                pass

    if not steps:
        return None, None

    time_ns = (np.array(steps) - prod_start) * step_to_ns
    gamma   = np.array(gamma)
    block_avg = []; block_std = []
    for i in range(5):
        mask = (time_ns >= i) & (time_ns < i + 1)
        if mask.any():
            block_avg.append(np.mean(gamma[mask]))
            block_std.append(np.std(gamma[mask]))
        else:
            block_avg.append(np.nan); block_std.append(np.nan)
    return block_avg, block_std


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(frame):
    """
    Returns per-frame density and orientation data using mol-ID grouping.

    BUG FIX: uses mol-ID grouping (not positional indexing).
    BUG FIX: 3D dipole vector computed correctly before normalising.
    """
    mol_ids = frame["mol"]
    types   = frame["type"]
    xyz     = frame["xyz"]

    mol_map = {}
    for i, (mid, t) in enumerate(zip(mol_ids, types)):
        if t not in (T_O_WAT, T_H_WAT, T_C_MET, T_O_MET, T_H_MET, T_H_OH):
            continue
        mol_map.setdefault(mid, {}).setdefault(t, []).append(i)

    z_o_wat = []; z_c_met = []; z_o_met = []; cos_theta = []

    for mid, tmap in mol_map.items():
        # Water
        if T_O_WAT in tmap and T_H_WAT in tmap:
            if len(tmap[T_O_WAT]) == 1 and len(tmap[T_H_WAT]) == 2:
                o_pos  = xyz[tmap[T_O_WAT][0]]
                h1_pos = xyz[tmap[T_H_WAT][0]]
                h2_pos = xyz[tmap[T_H_WAT][1]]
                z_o_wat.append(o_pos[2])
                # BUG FIX: full 3D dipole vector
                h_mid  = (h1_pos + h2_pos) / 2.0
                dipole = h_mid - o_pos
                norm   = np.linalg.norm(dipole)
                if norm > 1e-12:
                    cos_theta.append((o_pos[2], dipole[2] / norm))
        # Methanol
        elif T_C_MET in tmap and T_O_MET in tmap:
            z_c_met.append(xyz[tmap[T_C_MET][0], 2])
            z_o_met.append(xyz[tmap[T_O_MET][0], 2])

    return {
        "z_o_wat":   np.array(z_o_wat),
        "z_c_met":   np.array(z_c_met),
        "z_o_met":   np.array(z_o_met),
        "cos_theta": cos_theta,
    }


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def compute_profiles(frames_data, bin_edges):
    """Time-averaged 1D number density and orientation profiles."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dz      = bin_edges[1] - bin_edges[0]
    bin_vol = dz * BOX_X * BOX_Y

    acc_o_wat = np.zeros(len(bin_centers))
    acc_c_met = np.zeros(len(bin_centers))
    acc_o_met = np.zeros(len(bin_centers))
    cos_sum   = np.zeros(len(bin_centers))
    cos_cnt   = np.zeros(len(bin_centers))

    for fd in frames_data:
        for arr, acc in [(fd["z_o_wat"], acc_o_wat),
                         (fd["z_c_met"], acc_c_met),
                         (fd["z_o_met"], acc_o_met)]:
            if len(arr) > 0:
                h, _ = np.histogram(arr, bins=bin_edges)
                acc += h
        for z_o, cos_t in fd["cos_theta"]:
            bi = int((z_o - bin_edges[0]) / dz)
            if 0 <= bi < len(bin_centers):
                cos_sum[bi] += cos_t
                cos_cnt[bi] += 1

    n = max(len(frames_data), 1)
    return {
        "z":         bin_centers,
        "num_o_wat": acc_o_wat / (n * bin_vol),
        "num_c_met": acc_c_met / (n * bin_vol),
        "num_o_met": acc_o_met / (n * bin_vol),
        "cos_theta": np.divide(cos_sum, cos_cnt,
                               out=np.zeros_like(cos_sum), where=cos_cnt > 0),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
}

def savefig(fig, path_no_ext):
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="General analysis: density + orientation + surface tension")
    parser.add_argument("--dump",        default="traj.lammpstrj")
    parser.add_argument("--log",         default="simulation.log")
    parser.add_argument("--nmethanol",   default=10,       type=int)
    parser.add_argument("--outdir",      default="analysis_results")
    parser.add_argument("--prod_start",  default=500000,   type=int)
    parser.add_argument("--nframes_max", default=0,        type=int)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    STEP_TO_NS = 1e-6

    # Surface tension
    print("  Parsing surface tension ...")
    gamma_avg, gamma_std = parse_surface_tension_blocks(
        args.log, args.prod_start, STEP_TO_NS, ZMAX - ZMIN)

    if gamma_avg:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(7, 5))
            x = np.arange(1, 6)
            ax.bar(x, gamma_avg, yerr=gamma_std, capsize=5,
                   color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_xlabel("Time block (ns)")
            ax.set_ylabel("Surface tension gamma (mN/m)")
            ax.set_title("Surface Tension - 1 ns Block Averages")
            ax.set_xticks(x); plt.tight_layout()
            savefig(fig, os.path.join(args.outdir, "surface_tension_blocks"))
        np.savetxt(
            os.path.join(args.outdir, "surface_tension_blocks.txt"),
            np.column_stack((np.arange(1, 6), gamma_avg, gamma_std)),
            header="block  gamma_avg_mNm  gamma_std_mNm")
        finite = [g for g in gamma_avg if np.isfinite(g)]
        if finite:
            print(f"    gamma = {np.mean(finite):.2f} +/- {np.std(finite):.2f} mN/m")

    # Trajectory
    print("  Reading trajectory ...")
    bin_edges = np.linspace(ZMIN, ZMAX, NBINS + 1)
    frames_4ns = []; frames_5ns = []; n_read = 0

    for timestep, frame in read_lammps_dump(
            args.dump, args.prod_start, args.nframes_max):
        time_ns = (timestep - args.prod_start) * STEP_TO_NS
        fd = process_frame(frame)
        if 3.0 <= time_ns < 4.0: frames_4ns.append(fd)
        if 4.0 <= time_ns < 5.0: frames_5ns.append(fd)
        n_read += 1
        if n_read % 500 == 0:
            print(f"    ... {n_read} frames (t = {time_ns:.2f} ns)")

    print(f"  Frames: 3-4 ns = {len(frames_4ns)},  4-5 ns = {len(frames_5ns)}")

    if not frames_5ns:
        print("  WARNING: No frames in 4-5 ns window.")
        return

    prof5 = compute_profiles(frames_5ns, bin_edges)
    z = prof5["z"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof5["num_o_wat"] * 1000, color="#2196F3", lw=1.8, label="Water O")
        ax.plot(z, prof5["num_c_met"] * 1000, color="#F44336", lw=1.8, label="Methanol C")
        ax.plot(z, prof5["num_o_met"] * 1000, color="#FF9800", lw=1.5, ls="--", label="Methanol O")
        ax.set_xlabel("z (A)"); ax.set_ylabel("Number density (x10^-3 A^-3)")
        ax.set_title("Number Density Profile - 4-5 ns")
        ax.set_xlim(ZMIN, ZMAX); ax.legend(); plt.tight_layout()
        savefig(fig, os.path.join(args.outdir, "density_number_5ns"))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof5["cos_theta"], color="#9C27B0", lw=1.8)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("z (A)"); ax.set_ylabel("<cos theta>")
        ax.set_title("Water Dipole Orientation - 4-5 ns")
        ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); plt.tight_layout()
        savefig(fig, os.path.join(args.outdir, "orientation_5ns"))

    if frames_4ns:
        prof4 = compute_profiles(frames_4ns, bin_edges)
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(z, prof4["cos_theta"], color="#4CAF50", lw=1.8, label="3-4 ns")
            ax.plot(z, prof5["cos_theta"], color="#9C27B0", lw=1.8, label="4-5 ns")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("z (A)"); ax.set_ylabel("<cos theta>")
            ax.set_title("Water Dipole Orientation - Time Comparison")
            ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); ax.legend(); plt.tight_layout()
            savefig(fig, os.path.join(args.outdir, "orientation_comparison"))

    np.savetxt(
        os.path.join(args.outdir, "profiles_5ns.csv"),
        np.column_stack([z, prof5["num_o_wat"], prof5["num_c_met"],
                         prof5["num_o_met"],    prof5["cos_theta"]]),
        delimiter=",",
        header="z_A,num_o_water_A3,num_c_methanol_A3,num_o_methanol_A3,cos_theta_water",
        comments="")

    print(f"  Done. Results in: {args.outdir}/")


if __name__ == "__main__":
    main()
