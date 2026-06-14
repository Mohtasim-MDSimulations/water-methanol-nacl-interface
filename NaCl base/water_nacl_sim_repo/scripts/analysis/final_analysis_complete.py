#!/usr/bin/env python3
"""
final_analysis_complete.py
==========================
Complete analysis for the water-NaCl liquid-vapour interface simulation.

Produces (all as PNG + PDF):
  Surface tension:
    surface_tension_bars_raw.pdf/png       – block-averaged raw γ (1 ns windows)

  Ion density profiles (4-5 ns window):
    density_na_5ns_line.pdf/png
    density_cl_5ns_line.pdf/png
    density_na_cl_5ns_line.pdf/png         – Na+ and Cl- overlaid
    density_na_cl_5ns_bars.pdf/png         – sampled bar chart version

  Water orientation profiles:
    orientation_4ns_line.pdf/png           – 3-4 ns window
    orientation_4ns_bars.pdf/png
    orientation_5ns_line.pdf/png           – 4-5 ns window
    orientation_5ns_bars.pdf/png
    orientation_comparison_line.pdf/png    – both windows overlaid
    orientation_comparison_bars.pdf/png

  Data files:
    surface_tension_blocks.txt
    ion_density_profiles_5ns.csv
    water_orientation_4ns.txt
    water_orientation_5ns.txt

BUG FIXES vs original:
  1. Surface tension scaling REMOVED. The original script divided raw γ by
     mean_raw and multiplied by 72.0 mN/m, forcing the result to equal the
     textbook water value regardless of what the simulation actually produced.
     This is not a valid calibration — it masks force-field or setup errors.
     Fix: report raw γ directly. The correct SPC/Fw bulk value (~63-65 mN/m)
     naturally emerges from a well-set-up simulation.

  2. Timestep offset ADDED. production.in now uses reset_timestep 0, so
     production frames genuinely start at step 0. The analysis uses
     PROD_START = 0 and correctly maps step × 1e-6 → ns. If you run an
     older trajectory that did NOT reset the timestep, pass --prod_start
     with the actual first production step.

  3. Mol-ID-based molecule grouping replaces positional grouping.
     Original assumed atoms in the dump file are sorted by mol-ID then
     atom-type. Fixed with explicit mol-ID dictionary grouping.

Usage:
  python3 final_analysis_complete.py [--traj FILE] [--st FILE]
                                     [--outdir DIR]
                                     [--prod_start 0]
                                     [--nframes_max 0]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, argparse, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

# Atom-type constants
T_O_WAT = 1; T_H_WAT = 2; T_NA = 3; T_CL = 4
MASSES  = {T_O_WAT: 15.9994, T_H_WAT: 1.008, T_NA: 22.98977, T_CL: 35.453}

BOX_X, BOX_Y  = 60.0, 60.0
ZMIN,  ZMAX   = 0.0, 200.0
NBINS         = 200
STEP_TO_NS    = 1e-6   # 1 fs timestep


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_lammps_dump(filename, prod_start, nframes_max=0):
    """
    Generator yielding (timestep, frame_dict) for production frames.

    frame_dict keys:
      mol   – ndarray int   molecule ID per atom
      type  – ndarray int   atom type per atom
      xyz   – ndarray Nx3   coordinates

    BUG FIX: mol-ID read from dump; no positional indexing.
    """
    n_yielded = 0
    open_fn   = open

    with open_fn(filename) as fh:
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
            idx_type = cmap.get("type",  2)
            idx_x    = cmap.get("x",     3)
            idx_y    = cmap.get("y",     4)
            idx_z    = cmap.get("z",     5)
            has_mol  = (idx_mol >= 0)

            mol_arr  = np.zeros(natoms, dtype=int)
            type_arr = np.zeros(natoms, dtype=int)
            xyz_arr  = np.zeros((natoms, 3))

            for i in range(natoms):
                parts         = fh.readline().split()
                mol_arr[i]    = int(float(parts[idx_mol])) if has_mol else i
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


def read_surface_tension(filename):
    """
    Read surface_tension.dat written by LAMMPS fix ave/time.
    Returns (time_ns_array, gamma_mNm_array).
    """
    if not os.path.exists(filename):
        return None, None
    steps = []; gamma = []
    with open(filename) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                steps.append(float(parts[0]))
                gamma.append(float(parts[1]))
            except ValueError:
                pass
    if not steps:
        return None, None
    return np.array(steps) * STEP_TO_NS, np.array(gamma)


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(frame):
    """
    Extract ion density and water orientation data using mol-ID grouping.

    BUG FIX: mol-ID grouping replaces positional atom indexing.
    """
    mol_ids = frame["mol"]
    types   = frame["type"]
    xyz     = frame["xyz"]

    # Group atoms by mol-ID
    mol_map = {}
    for i, (mid, t) in enumerate(zip(mol_ids, types)):
        mol_map.setdefault(mid, {}).setdefault(t, []).append(i)

    z_na  = []
    z_cl  = []
    cos_theta_water = []   # (z_O, cos_theta) pairs

    for mid, tmap in mol_map.items():
        # Na+ ion
        if T_NA in tmap and len(tmap.get(T_NA, [])) == 1:
            if T_O_WAT not in tmap and T_H_WAT not in tmap:
                z_na.append(xyz[tmap[T_NA][0], 2])

        # Cl- ion
        elif T_CL in tmap and len(tmap.get(T_CL, [])) == 1:
            if T_O_WAT not in tmap and T_H_WAT not in tmap:
                z_cl.append(xyz[tmap[T_CL][0], 2])

        # Water molecule
        elif T_O_WAT in tmap and T_H_WAT in tmap:
            if len(tmap[T_O_WAT]) == 1 and len(tmap[T_H_WAT]) == 2:
                o_pos  = xyz[tmap[T_O_WAT][0]]
                h1_pos = xyz[tmap[T_H_WAT][0]]
                h2_pos = xyz[tmap[T_H_WAT][1]]
                # Full 3D dipole vector (O → midpoint of H's)
                h_mid  = (h1_pos + h2_pos) / 2.0
                dipole = h_mid - o_pos
                norm   = np.linalg.norm(dipole)
                if norm > 1e-12:
                    cos_theta_water.append((o_pos[2], dipole[2] / norm))

    return {
        "z_na":             np.array(z_na),
        "z_cl":             np.array(z_cl),
        "cos_theta_water":  cos_theta_water,
    }


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def compute_ion_density(frames_data, bin_edges):
    """Time-averaged ion number density profiles."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dz      = bin_edges[1] - bin_edges[0]
    bin_vol = dz * BOX_X * BOX_Y

    acc_na = np.zeros(len(bin_centers))
    acc_cl = np.zeros(len(bin_centers))

    for fd in frames_data:
        if len(fd["z_na"]) > 0:
            h, _ = np.histogram(fd["z_na"], bins=bin_edges); acc_na += h
        if len(fd["z_cl"]) > 0:
            h, _ = np.histogram(fd["z_cl"], bins=bin_edges); acc_cl += h

    n = max(len(frames_data), 1)
    return {
        "z":     bin_centers,
        "na":    acc_na / (n * bin_vol),
        "cl":    acc_cl / (n * bin_vol),
    }


def compute_orientation(frames_data, bin_edges):
    """Time-averaged water dipole orientation profile."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dz      = bin_edges[1] - bin_edges[0]
    cos_sum = np.zeros(len(bin_centers))
    cos_cnt = np.zeros(len(bin_centers))

    for fd in frames_data:
        for z_o, cos_t in fd["cos_theta_water"]:
            bi = int((z_o - bin_edges[0]) / dz)
            if 0 <= bi < len(bin_centers):
                cos_sum[bi] += cos_t
                cos_cnt[bi] += 1

    return {
        "z":         bin_centers,
        "cos_theta": np.divide(cos_sum, cos_cnt,
                               out=np.zeros_like(cos_sum), where=cos_cnt > 0),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
}

def savefig(fig, path_no_ext):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_no_ext}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Complete analysis for water-NaCl interface")
    parser.add_argument("--traj",        default="production.lammpstrj")
    parser.add_argument("--st",          default="surface_tension.dat")
    parser.add_argument("--outdir",      default="results")
    parser.add_argument("--prod_start",  default=0,   type=int,
                        help="First production timestep (0 if reset_timestep 0 was used)")
    parser.add_argument("--nframes_max", default=0,   type=int)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    p = lambda name: os.path.join(args.outdir, name)

    # ── 1. Surface tension ────────────────────────────────────────────────────
    print("  Processing surface tension ...")
    time_ns_st, gamma_raw = read_surface_tension(args.st)

    if gamma_raw is not None:
        # BUG FIX 1: report raw gamma, no forced scaling
        block_avg = []; block_std = []
        for i in range(5):
            mask = (time_ns_st >= i) & (time_ns_st < i + 1)
            if mask.any():
                block_avg.append(np.mean(gamma_raw[mask]))
                block_std.append(np.std(gamma_raw[mask]))
            else:
                block_avg.append(np.nan); block_std.append(np.nan)

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(1, 6)
            ax.bar(x, block_avg, yerr=block_std, capsize=5,
                   color="steelblue", edgecolor="black", alpha=0.8)
            ax.axhline(63.0, color="red", ls="--", lw=1.2,
                       label="SPC/Fw bulk ref. (~63 mN/m)")
            ax.set_xlabel("Time block (ns)")
            ax.set_ylabel("Surface tension γ (mN/m)")
            ax.set_title("Block-Averaged Surface Tension (Raw)")
            ax.set_xticks(x); ax.legend(); plt.tight_layout()
            savefig(fig, p("surface_tension_bars_raw"))

        np.savetxt(p("surface_tension_blocks.txt"),
                   np.column_stack((np.arange(1, 6), block_avg, block_std)),
                   header="block  gamma_avg_mNm  gamma_std_mNm")
        finite = [g for g in block_avg if np.isfinite(g)]
        if finite:
            print(f"    γ = {np.mean(finite):.2f} ± {np.std(finite):.2f} mN/m "
                  f"({len(finite)} blocks)")
    else:
        print("    surface_tension.dat not found — skipping γ plot")

    # ── 2. Trajectory ─────────────────────────────────────────────────────────
    print("  Reading trajectory ...")
    bin_edges = np.linspace(ZMIN, ZMAX, NBINS + 1)

    frames_4ns = []   # 3–4 ns
    frames_5ns = []   # 4–5 ns
    n_read = 0

    for timestep, frame in read_lammps_dump(
            args.traj, args.prod_start, args.nframes_max):
        # BUG FIX 2: time_ns uses prod_start offset
        time_ns = (timestep - args.prod_start) * STEP_TO_NS
        fd = process_frame(frame)
        if 3.0 <= time_ns < 4.0: frames_4ns.append(fd)
        if 4.0 <= time_ns < 5.0: frames_5ns.append(fd)
        n_read += 1
        if n_read % 50 == 0:
            print(f"    ... {n_read} frames read (t = {time_ns:.2f} ns)")

    print(f"  Frames: 3–4 ns = {len(frames_4ns)},  4–5 ns = {len(frames_5ns)}")

    if not frames_5ns:
        print("  WARNING: No frames in 4-5 ns window. "
              "Check --prod_start and trajectory length.")
        sys.exit(1)

    # ── 3. Ion density profiles (4–5 ns) ──────────────────────────────────────
    print("  Computing ion density profiles ...")
    prof_ion = compute_ion_density(frames_5ns, bin_edges)
    z = prof_ion["z"]

    with plt.rc_context(STYLE):
        # Na+ line
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof_ion["na"] * 1000, color="#E67E22", lw=2, label="Na⁺")
        ax.set_xlabel("z (Å)"); ax.set_ylabel("Number density (×10⁻³ Å⁻³)")
        ax.set_title("Na⁺ Density – 4–5 ns"); ax.legend()
        ax.set_xlim(ZMIN, ZMAX); plt.tight_layout()
        savefig(fig, p("density_na_5ns_line"))

        # Cl- line
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof_ion["cl"] * 1000, color="#27AE60", lw=2, label="Cl⁻")
        ax.set_xlabel("z (Å)"); ax.set_ylabel("Number density (×10⁻³ Å⁻³)")
        ax.set_title("Cl⁻ Density – 4–5 ns"); ax.legend()
        ax.set_xlim(ZMIN, ZMAX); plt.tight_layout()
        savefig(fig, p("density_cl_5ns_line"))

        # Na+ + Cl- overlaid (line)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, prof_ion["na"] * 1000, color="#E67E22", lw=2, label="Na⁺")
        ax.plot(z, prof_ion["cl"] * 1000, color="#27AE60", lw=2, label="Cl⁻")
        ax.set_xlabel("z (Å)"); ax.set_ylabel("Number density (×10⁻³ Å⁻³)")
        ax.set_title("Ion Number Densities – 4–5 ns"); ax.legend()
        ax.set_xlim(ZMIN, ZMAX); plt.tight_layout()
        savefig(fig, p("density_na_cl_5ns_line"))

        # Na+ + Cl- bar (sampled)
        step = max(1, NBINS // 20)
        z_s  = z[::step]; na_s = prof_ion["na"][::step] * 1000
        cl_s = prof_ion["cl"][::step] * 1000
        x_pos = np.arange(len(z_s)); w = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x_pos - w/2, na_s, w, label="Na⁺", color="#E67E22", alpha=0.8)
        ax.bar(x_pos + w/2, cl_s, w, label="Cl⁻", color="#27AE60", alpha=0.8)
        ax.set_xlabel("z (Å)"); ax.set_ylabel("Number density (×10⁻³ Å⁻³)")
        ax.set_title("Ion Densities (Bar) – 4–5 ns"); ax.legend()
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{zv:.0f}" for zv in z_s], rotation=45)
        plt.tight_layout(); savefig(fig, p("density_na_cl_5ns_bars"))

    # Save CSV
    np.savetxt(
        p("ion_density_profiles_5ns.csv"),
        np.column_stack([z, prof_ion["na"], prof_ion["cl"]]),
        delimiter=",",
        header="z_A,num_na_A3,num_cl_A3", comments="")

    print("    Ion density plots saved.")

    # ── 4. Water orientation profiles ─────────────────────────────────────────
    print("  Computing water orientation profiles ...")

    orient_data = {}
    for (t0, t1), frames, label in [
            ((3, 4), frames_4ns, "4ns"),
            ((4, 5), frames_5ns, "5ns")]:
        if not frames:
            print(f"    No frames in {t0}–{t1} ns window — skipping")
            continue
        prof = compute_orientation(frames, bin_edges)
        orient_data[label] = prof

        with plt.rc_context(STYLE):
            # Line
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(prof["z"], prof["cos_theta"], color="#9C27B0", lw=2)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("z (Å)"); ax.set_ylabel("⟨cos θ⟩")
            ax.set_title(f"Water Dipole Orientation – {t0}–{t1} ns")
            ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); plt.tight_layout()
            savefig(fig, p(f"orientation_{label}_line"))

            # Bar (sampled)
            z_s  = prof["z"][::step]; cos_s = prof["cos_theta"][::step]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(z_s, cos_s, width=step * (ZMAX - ZMIN) / NBINS * 0.8,
                   color="#9C27B0", alpha=0.75, edgecolor="black")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("z (Å)"); ax.set_ylabel("⟨cos θ⟩")
            ax.set_title(f"Water Dipole Orientation (Bar) – {t0}–{t1} ns")
            ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); plt.tight_layout()
            savefig(fig, p(f"orientation_{label}_bars"))

        np.savetxt(
            p(f"water_orientation_{label}.txt"),
            np.column_stack([prof["z"], prof["cos_theta"]]),
            header="z_A  cos_theta")
        print(f"    Orientation plots for {t0}–{t1} ns saved.")

    # Comparison plots
    if len(orient_data) == 2:
        with plt.rc_context(STYLE):
            # Line comparison
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(orient_data["4ns"]["z"], orient_data["4ns"]["cos_theta"],
                    color="#4CAF50", lw=2, label="3–4 ns")
            ax.plot(orient_data["5ns"]["z"], orient_data["5ns"]["cos_theta"],
                    color="#9C27B0", lw=2, label="4–5 ns")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("z (Å)"); ax.set_ylabel("⟨cos θ⟩")
            ax.set_title("Water Dipole Orientation – Time Comparison")
            ax.set_xlim(ZMIN, ZMAX); ax.set_ylim(-1, 1); ax.legend()
            plt.tight_layout(); savefig(fig, p("orientation_comparison_line"))

            # Bar comparison
            z_s   = orient_data["4ns"]["z"][::step]
            cos4s = orient_data["4ns"]["cos_theta"][::step]
            cos5s = orient_data["5ns"]["cos_theta"][::step]
            x_pos = np.arange(len(z_s))
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.bar(x_pos - w/2, cos4s, w, label="3–4 ns", color="#4CAF50", alpha=0.8)
            ax.bar(x_pos + w/2, cos5s, w, label="4–5 ns", color="#9C27B0", alpha=0.8)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel("z (Å)"); ax.set_ylabel("⟨cos θ⟩")
            ax.set_title("Water Dipole Orientation Comparison (Bar)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{zv:.0f}" for zv in z_s], rotation=45)
            ax.legend(); plt.tight_layout()
            savefig(fig, p("orientation_comparison_bars"))

        print("    Comparison plots saved.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ALL ANALYSIS COMPLETE")
    print(f"  Output directory: {args.outdir}/")
    print(f"{'='*60}")
    files = sorted(f for f in os.listdir(args.outdir)
                   if f.endswith((".pdf", ".png", ".txt", ".csv")))
    for fn in files:
        print(f"    {fn}")


if __name__ == "__main__":
    main()
