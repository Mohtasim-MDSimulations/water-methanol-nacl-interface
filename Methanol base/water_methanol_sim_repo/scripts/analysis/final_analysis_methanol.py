#!/usr/bin/env python3
"""
final_analysis_methanol.py
==========================
Methanol-focused analysis for the water-methanol liquid-vapour interface.

Produces:
  - Surface tension bar graph (1 ns blocks)
  - Methanol number density profile (C and O sites)
  - Methanol mass density profile
  - Water dipole orientation profiles at 3-4 ns and 4-5 ns
  - Methanol O-H orientation profile
  - All profiles saved as text/CSV

BUG FIXES vs original:
  1. Positional indexing removed: original used a fixed byte offset
     (TOTAL_WATER_ATOMS) to locate methanol atoms, which is only correct
     if LAMMPS writes atoms in exact creation order — not guaranteed under
     MPI. Fixed by grouping atoms by mol-ID from the dump.

  2. Methanol O-H orientation added: the original script computed water
     orientation but did not compute the analogous methanol O-H vector
     orientation, which is the primary interfacial observable for methanol.

  3. CLI arguments: n_methanol, dump path, log path are parameters rather
     than hardcoded constants.

Usage:
  python3 final_analysis_methanol.py [--dump traj.lammpstrj]
                                     [--log simulation.log]
                                     [--nmethanol 10]
                                     [--outdir final_analysis_methanol]
                                     [--prod_start 500000]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

# Atom-type constants
T_O_WAT = 1; T_H_WAT = 2
T_C_MET = 3; T_O_MET = 4; T_H_MET = 5; T_H_OH  = 6

MASSES = {
    T_C_MET: 12.0110, T_O_MET: 15.9994,
    T_H_MET: 1.00794, T_H_OH:  1.00794,
    T_O_WAT: 15.9994, T_H_WAT: 1.00794,
}
METHANOL_MOLAR_MASS = 12.0110 + 15.9994 + 4 * 1.00794   # 32.042 g/mol

BOX_X, BOX_Y = 60.0, 60.0
ZMIN,  ZMAX  = 0.0, 200.0
NBINS        = 200
STEP_TO_NS   = 1e-6


# ---------------------------------------------------------------------------
# I/O  (shared with analysis.py)
# ---------------------------------------------------------------------------

def read_lammps_dump(filename, prod_start, nframes_max=0):
    """
    Generator yielding (timestep, frame_dict) for production frames.
    Reads mol column; no positional atom indexing.
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


def parse_surface_tension_blocks(logfile, prod_start, zlen):
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
                pxx, pyy, pzz = float(parts[6]), float(parts[7]), float(parts[8])
                conv = 0.0101325 * zlen / 2.0
                steps.append(step)
                gamma.append(conv * (pzz - (pxx + pyy) / 2.0))
            except (ValueError, IndexError):
                pass
    if not steps:
        return None, None
    time_ns = (np.array(steps) - prod_start) * STEP_TO_NS
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
    Extract methanol and water data from one frame using mol-ID grouping.

    Returns:
      z_c_met        - z of methanol C atoms
      z_o_met        - z of methanol O atoms
      z_com_met      - z of methanol centre-of-mass
      mass_met       - total mass of each methanol molecule
      cos_oh_met     - list of (z_O, cos_theta) for methanol O-H vectors
      cos_theta_wat  - list of (z_O, cos_theta) for water dipole vectors
    """
    mol_ids = frame["mol"]
    types   = frame["type"]
    xyz     = frame["xyz"]

    mol_map = {}
    for i, (mid, t) in enumerate(zip(mol_ids, types)):
        mol_map.setdefault(mid, {}).setdefault(t, []).append(i)

    z_c_met   = []; z_o_met    = []
    z_com_met = []; mass_met   = []
    cos_oh_met    = []    # methanol O-H orientation
    cos_theta_wat = []    # water dipole orientation

    for mid, tmap in mol_map.items():
        # ── Methanol ─────────────────────────────────────────────────────────
        if (T_C_MET in tmap and T_O_MET in tmap
                and T_H_OH in tmap and T_H_MET in tmap):
            c_idx    = tmap[T_C_MET][0]
            o_idx    = tmap[T_O_MET][0]
            h_oh_idx = tmap[T_H_OH][0]
            h_met_idx = tmap[T_H_MET]   # list of up to 3 indices

            c_pos   = xyz[c_idx]
            o_pos   = xyz[o_idx]
            h_oh_pos = xyz[h_oh_idx]

            z_c_met.append(c_pos[2])
            z_o_met.append(o_pos[2])

            # Centre-of-mass
            total_mass = (MASSES[T_C_MET] + MASSES[T_O_MET] + MASSES[T_H_OH]
                          + len(h_met_idx) * MASSES[T_H_MET])
            com_z = (c_pos[2]   * MASSES[T_C_MET]
                     + o_pos[2] * MASSES[T_O_MET]
                     + h_oh_pos[2] * MASSES[T_H_OH]
                     + sum(xyz[hi, 2] for hi in h_met_idx) * MASSES[T_H_MET]
                     ) / total_mass
            z_com_met.append(com_z)
            mass_met.append(total_mass)

            # O-H orientation: vector from O to hydroxyl H
            oh_vec = h_oh_pos - o_pos
            norm   = np.linalg.norm(oh_vec)
            if norm > 1e-12:
                cos_oh_met.append((o_pos[2], oh_vec[2] / norm))

        # ── Water ─────────────────────────────────────────────────────────────
        elif T_O_WAT in tmap and T_H_WAT in tmap:
            if len(tmap[T_O_WAT]) == 1 and len(tmap[T_H_WAT]) == 2:
                o_pos  = xyz[tmap[T_O_WAT][0]]
                h1_pos = xyz[tmap[T_H_WAT][0]]
                h2_pos = xyz[tmap[T_H_WAT][1]]
                h_mid  = (h1_pos + h2_pos) / 2.0
                dipole = h_mid - o_pos
                norm   = np.linalg.norm(dipole)
                if norm > 1e-12:
                    cos_theta_wat.append((o_pos[2], dipole[2] / norm))

    return {
        "z_c_met":       np.array(z_c_met),
        "z_o_met":       np.array(z_o_met),
        "z_com_met":     np.array(z_com_met),
        "mass_met":      np.array(mass_met),
        "cos_oh_met":    cos_oh_met,
        "cos_theta_wat": cos_theta_wat,
    }


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def compute_profiles(frames_data, bin_edges):
    """Time-averaged density and orientation profiles."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dz      = bin_edges[1] - bin_edges[0]
    bin_vol = dz * BOX_X * BOX_Y

    acc_c   = np.zeros(len(bin_centers))
    acc_o   = np.zeros(len(bin_centers))
    mass_acc = np.zeros(len(bin_centers))

    cos_oh_sum  = np.zeros(len(bin_centers)); cos_oh_cnt  = np.zeros(len(bin_centers))
    cos_wat_sum = np.zeros(len(bin_centers)); cos_wat_cnt = np.zeros(len(bin_centers))

    for fd in frames_data:
        if len(fd["z_c_met"]) > 0:
            h, _ = np.histogram(fd["z_c_met"], bins=bin_edges); acc_c += h
        if len(fd["z_o_met"]) > 0:
            h, _ = np.histogram(fd["z_o_met"], bins=bin_edges); acc_o += h
        if len(fd["z_com_met"]) > 0:
            h, _ = np.histogram(fd["z_com_met"], bins=bin_edges,
                                 weights=fd["mass_met"])
            mass_acc += h

        for z_o, cos_t in fd["cos_oh_met"]:
            bi = int((z_o - bin_edges[0]) / dz)
            if 0 <= bi < len(bin_centers):
                cos_oh_sum[bi] += cos_t; cos_oh_cnt[bi] += 1

        for z_o, cos_t in fd["cos_theta_wat"]:
            bi = int((z_o - bin_edges[0]) / dz)
            if 0 <= bi < len(bin_centers):
                cos_wat_sum[bi] += cos_t; cos_wat_cnt[bi] += 1

    n = max(len(frames_data), 1)
    # mass density: g/cm³  (1 amu/Å³ = 1.660539 g/cm³)
    mass_density = mass_acc * 1.660539 / (n * bin_vol)

    return {
        "z":            bin_centers,
        "num_c":        acc_c / (n * bin_vol),
        "num_o":        acc_o / (n * bin_vol),
        "mass_density": mass_density,
        "cos_oh_met":   np.divide(cos_oh_sum,  cos_oh_cnt,
                                  out=np.zeros_like(cos_oh_sum),  where=cos_oh_cnt  > 0),
        "cos_wat":      np.divide(cos_wat_sum, cos_wat_cnt,
                                  out=np.zeros_like(cos_wat_sum), where=cos_wat_cnt > 0),
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
        description="Methanol-focused analysis: density + O-H orientation + surface tension")
    parser.add_argument("--dump",       default="traj.lammpstrj")
    parser.add_argument("--log",        default="simulation.log")
    parser.add_argument("--nmethanol",  default=10,     type=int)
    parser.add_argument("--outdir",     default="final_analysis_methanol")
    parser.add_argument("--prod_start", default=500000, type=int)
    parser.add_argument("--nframes_max",default=0,      type=int)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Surface tension ───────────────────────────────────────────────────────
    print("  Parsing surface tension ...")
    gamma_avg, gamma_std = parse_surface_tension_blocks(
        args.log, args.prod_start, ZMAX - ZMIN)

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

    # ── Trajectory ────────────────────────────────────────────────────────────
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
        print("  WARNING: No frames in 4-5 ns window. Check --prod_start.")
        sys.exit(1)

    prof5 = compute_profiles(frames_5ns, bin_edges)
    z = prof5["z"]

    with plt.rc_context(STYLE):
        # Methanol number density
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof5["num_c"] * 1000, color="#2196F3", lw=1.8, label="Methanol C")
        ax.plot(z, prof5["num_o"] * 1000, color="#F44336", lw=1.8, label="Methanol O")
        ax.set_xlabel("z (A)"); ax.set_ylabel("Number density (x10^-3 A^-3)")
        ax.set_title("Methanol Number Density - 4-5 ns")
        ax.set_xlim(ZMIN, ZMAX); ax.legend(); plt.tight_layout()
        savefig(fig, os.path.join(args.outdir, "methanol_number_density_5ns"))

        # Methanol mass density
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof5["mass_density"], color="#4CAF50", lw=1.8)
        ax.set_xlabel("z (A)"); ax.set_ylabel("Mass density (g/cm^3)")
        ax.set_title("Methanol Mass Density - 4-5 ns")
        ax.set_xlim(ZMIN, ZMAX); plt.tight_layout()
        savefig(fig, os.path.join(args.outdir, "methanol_mass_density_5ns"))

        # Methanol O-H orientation
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof5["cos_oh_met"], color="#E91E63", lw=1.8)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("z (A)"); ax.set_ylabel("<cos theta> (O-H vector)")
        ax.set_title("Methanol O-H Orientation - 4-5 ns")
        ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); plt.tight_layout()
        savefig(fig, os.path.join(args.outdir, "methanol_oh_orientation_5ns"))

        # Water orientation at 4-5 ns
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof5["cos_wat"], color="#9C27B0", lw=1.8)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("z (A)"); ax.set_ylabel("<cos theta> (dipole)")
        ax.set_title("Water Dipole Orientation - 4-5 ns")
        ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); plt.tight_layout()
        savefig(fig, os.path.join(args.outdir, "water_orientation_5ns"))

    # ── 3-4 ns window ─────────────────────────────────────────────────────────
    if frames_4ns:
        prof4 = compute_profiles(frames_4ns, bin_edges)
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(z, prof4["cos_wat"],      color="#4CAF50", lw=1.8, label="Water 3-4 ns")
            ax.plot(z, prof5["cos_wat"],      color="#9C27B0", lw=1.8, label="Water 4-5 ns")
            ax.plot(z, prof5["cos_oh_met"],   color="#E91E63", lw=1.8, ls="--",
                    label="Methanol O-H 4-5 ns")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("z (A)"); ax.set_ylabel("<cos theta>")
            ax.set_title("Orientation Summary - Water & Methanol")
            ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); ax.legend(); plt.tight_layout()
            savefig(fig, os.path.join(args.outdir, "orientation_summary"))

        np.savetxt(
            os.path.join(args.outdir, "orientation_4ns.txt"),
            np.column_stack((z, prof4["cos_wat"])),
            header="z_A  cos_theta_water")

    # ── CSV summary ───────────────────────────────────────────────────────────
    np.savetxt(
        os.path.join(args.outdir, "methanol_density_5ns.txt"),
        np.column_stack((z, prof5["num_c"], prof5["num_o"], prof5["mass_density"])),
        header="z(A)  num_c(A^-3)  num_o(A^-3)  mass(g/cm^3)")

    np.savetxt(
        os.path.join(args.outdir, "all_profiles_5ns.csv"),
        np.column_stack([z, prof5["num_c"],  prof5["num_o"],
                         prof5["mass_density"], prof5["cos_oh_met"],
                         prof5["cos_wat"]]),
        delimiter=",",
        header="z_A,num_C_methanol,num_O_methanol,mass_density_gcm3,"
               "cos_theta_OH_methanol,cos_theta_water",
        comments="")

    print(f"  Done. Results in: {args.outdir}/")


if __name__ == "__main__":
    main()
