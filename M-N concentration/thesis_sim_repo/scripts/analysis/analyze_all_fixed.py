#!/usr/bin/env python3
"""
analyze_all_fixed.py
====================
Multi-concentration comparison using the fixed analysis engine.
Generates individual plots for each system AND overlay comparison plots
across all four mixed concentrations (M10N10, M20N20, M50N50, M100N100).

Outputs (in --outdir, default: line_graph_fixed/):
  <outdir>/<sim_name>/   – per-system plots (PDF + PNG)
  <outdir>/comparisons/  – overlay comparison plots (PDF + PNG)
  <outdir>/summary.csv   – all profiles concatenated

Usage (run from the repo root or simulations/ parent directory):
  python3 scripts/analysis/analyze_all_fixed.py [--nframes 50]
          [--binwidth 1.0] [--outdir line_graph_fixed]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

# Import shared engine
sys.path.insert(0, os.path.dirname(__file__))
from analysis_fixed import (
    BOX_LX, BOX_LY, BOX_Z, PLOT_STYLE,
    T_O_WAT, T_NA, T_CL, METHANOL_TYPES, ION_TYPES,
    load_mol_map_from_data, read_frames, find_gds,
    density_profile, orientation_water, orientation_methanol,
    orientation_nacl_zonal, save_line_plot, save_na_cl_plot,
)

# ── Simulation registry ───────────────────────────────────────────────────────
SIM_DIRS = {
    "M10_N10":   "simulations/sim_M10_N10",
    "M20_N20":   "simulations/sim_M20_N20",
    "M50_N50":   "simulations/sim_M50_N50",
    "M100_N100": "simulations/sim_M100_N100",
}
DISPLAY = {
    "M10_N10":   "10 Methanol / 10 NaCl",
    "M20_N20":   "20 Methanol / 20 NaCl",
    "M50_N50":   "50 Methanol / 50 NaCl",
    "M100_N100": "100 Methanol / 100 NaCl",
}
COLORS = {
    "M10_N10":   "#1f77b4",
    "M20_N20":   "#ff7f0e",
    "M50_N50":   "#2ca02c",
    "M100_N100": "#d62728",
}
DEFAULT_NFRAMES  = 50
DEFAULT_BINWIDTH = 1.0
DEFAULT_OUTDIR   = "line_graph_fixed"


# ── Comparison plot helpers ───────────────────────────────────────────────────

def comparison_line(all_data, key, title, ylabel, filepath,
                    ylim=None, zero_line=False, nan_ok=False):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, d in all_data.items():
            y = d[key]; z = d["z"]
            if nan_ok:
                mask = ~np.isnan(y)
                ax.plot(z[mask], y[mask], "o", color=COLORS[name],
                        ms=3, alpha=0.75, label=DISPLAY[name])
            else:
                ax.plot(z, y, color=COLORS[name], lw=1.8, label=DISPLAY[name])
            z_bot, z_top = d["gds"]
            ax.axvline(z_bot, color=COLORS[name], lw=0.6, ls=":", alpha=0.5)
            ax.axvline(z_top, color=COLORS[name], lw=0.6, ls=":", alpha=0.5)
        if zero_line:
            ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Z (Å)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlim(0, BOX_Z)
        if ylim: ax.set_ylim(ylim)
        ax.legend()
        plt.tight_layout()
        fig.savefig(filepath + ".pdf", bbox_inches="tight", dpi=300)
        fig.savefig(filepath + ".png", bbox_inches="tight", dpi=300)
        plt.close(fig)


def comparison_na_cl(all_data, filepath):
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axes = axes.flatten()
        for ax, (name, d) in zip(axes, all_data.items()):
            ax.plot(d["z"], d["rho_na"], color="#E67E22", lw=1.5, label="Na⁺")
            ax.plot(d["z"], d["rho_cl"], color="#27AE60", lw=1.5, label="Cl⁻")
            ax.axvline(d["gds"][0], color="gray", lw=0.8, ls=":")
            ax.axvline(d["gds"][1], color="gray", lw=0.8, ls=":")
            ax.set_title(DISPLAY[name]); ax.legend()
            ax.set_xlabel("Z (Å)"); ax.set_ylabel("atoms/Å³")
        plt.suptitle("Na⁺ vs Cl⁻ Density – All Concentrations",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        fig.savefig(filepath + ".pdf", bbox_inches="tight", dpi=300)
        fig.savefig(filepath + ".png", bbox_inches="tight", dpi=300)
        plt.close(fig)


def summary_panel(all_data, filepath):
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        panels = [
            ("rho_meoh",  "Methanol Density",       "atoms/Å³",                       False, False, None),
            ("rho_nacl",  "NaCl Density",            "atoms/Å³",                       False, False, None),
            (None,        "Na⁺ vs Cl⁻",             "atoms/Å³",                       False, False, None),
            ("w_orient",  "Water Orientation",       r"$\langle\cos\theta\rangle$",    True,  False, (-1, 1)),
            ("m_orient",  "Methanol Orientation",    r"$\langle\cos\theta\rangle$",    True,  False, (-1, 1)),
            ("n_orient",  "NaCl Orientation",        r"$\langle\cos\theta\rangle$",    True,  True,  (-1, 1)),
        ]
        for ax, (key, title, ylabel, zero, nan_ok, ylim) in zip(axes.flatten(), panels):
            for name, d in all_data.items():
                z = d["z"]; color = COLORS[name]; label = DISPLAY[name]
                if key is None:
                    ax.plot(z, d["rho_na"], color=color, lw=1.2, ls="-",
                            label=f"{label} Na⁺")
                    ax.plot(z, d["rho_cl"], color=color, lw=1.2, ls="--",
                            label=f"{label} Cl⁻")
                else:
                    y = d[key]
                    if nan_ok:
                        mask = ~np.isnan(y)
                        ax.plot(z[mask], y[mask], "o", color=color,
                                ms=2.5, alpha=0.7, label=label)
                    else:
                        ax.plot(z, y, color=color, lw=1.4, label=label)
            if zero:
                ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Z (Å)"); ax.set_ylabel(ylabel)
            ax.set_xlim(0, BOX_Z)
            if ylim: ax.set_ylim(ylim)
            if key == "rho_meoh":
                ax.legend(fontsize=8)
        plt.suptitle("Interface Structure – All Concentrations",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        fig.savefig(filepath + ".pdf", bbox_inches="tight", dpi=300)
        fig.savefig(filepath + ".png", bbox_inches="tight", dpi=300)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nframes",  default=DEFAULT_NFRAMES,  type=int)
    parser.add_argument("--binwidth", default=DEFAULT_BINWIDTH, type=float)
    parser.add_argument("--outdir",   default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    base = os.getcwd()
    bw   = args.binwidth
    lz   = BOX_Z
    os.makedirs(args.outdir, exist_ok=True)

    all_data = {}

    for name, sub in SIM_DIRS.items():
        spath = os.path.join(base, sub)
        if not os.path.isdir(spath):
            print(f"  SKIP {name}: directory '{spath}' not found")
            continue

        dump = None
        for cand in ["traj.lammpstrj", "traj_combined.lammpstrj",
                     "traj_new.lammpstrj", "traj_continued.lammpstrj"]:
            p = os.path.join(spath, cand)
            if os.path.exists(p):
                dump = p
                break
        if dump is None:
            print(f"  SKIP {name}: no trajectory file found in '{spath}'")
            continue

        datafile = os.path.join(spath, "system.data")
        mol_map  = load_mol_map_from_data(datafile) if os.path.exists(datafile) else None
        print(f"\n  Processing {name}: {dump}")

        frames = read_frames(dump, args.nframes, mol_map=mol_map)

        z, rho_water = density_profile(frames, {T_O_WAT},      bw, BOX_LX, BOX_LY, lz)
        _, rho_meoh  = density_profile(frames, METHANOL_TYPES, bw, BOX_LX, BOX_LY, lz)
        _, rho_nacl  = density_profile(frames, ION_TYPES,      bw, BOX_LX, BOX_LY, lz)
        _, rho_na    = density_profile(frames, {T_NA},         bw, BOX_LX, BOX_LY, lz)
        _, rho_cl    = density_profile(frames, {T_CL},         bw, BOX_LX, BOX_LY, lz)

        z_bot, z_top = find_gds(z, rho_water, lz)
        gds = (z_bot, z_top)
        print(f"    GDS: {z_bot:.1f} – {z_top:.1f} Å")

        _, w_orient = orientation_water(frames, bw, lz)
        _, m_orient = orientation_methanol(frames, bw, lz)
        _, n_orient = orientation_nacl_zonal(frames, bw, lz, z_bot, z_top)

        # Individual system plots
        ind_dir = os.path.join(args.outdir, name)
        os.makedirs(ind_dir, exist_ok=True)
        p = lambda n: os.path.join(ind_dir, n)

        save_line_plot(z, rho_meoh, f"Methanol Density – {DISPLAY[name]}",
                       "Z (Å)", "atoms/Å³", p("01_methanol_density"),
                       color="#2196F3", gds=gds)
        save_line_plot(z, rho_nacl, f"NaCl Density – {DISPLAY[name]}",
                       "Z (Å)", "atoms/Å³", p("02_nacl_density"),
                       color="#9C27B0", gds=gds)
        save_na_cl_plot(z, rho_na, rho_cl,
                        f"Na⁺ vs Cl⁻ – {DISPLAY[name]}", p("03_na_vs_cl"), gds=gds)
        save_line_plot(z, w_orient, f"Water Orientation – {DISPLAY[name]}",
                       "Z (Å)", r"$\langle\cos\theta\rangle$",
                       p("04_water_orient"), color="#2196F3",
                       ylim=(-1, 1), zero_line=True, gds=gds)
        save_line_plot(z, m_orient, f"Methanol Orientation – {DISPLAY[name]}",
                       "Z (Å)", r"$\langle\cos\theta\rangle$",
                       p("05_methanol_orient"), color="#F44336",
                       ylim=(-1, 1), zero_line=True, gds=gds)
        save_line_plot(z, n_orient, f"NaCl Orientation – {DISPLAY[name]}",
                       "Z (Å)", r"$\langle\cos\theta\rangle$",
                       p("06_nacl_orient"), color="#4CAF50",
                       ylim=(-1, 1), zero_line=True, gds=gds, nan_ok=True)

        all_data[name] = dict(
            z=z, rho_water=rho_water, rho_meoh=rho_meoh,
            rho_nacl=rho_nacl, rho_na=rho_na, rho_cl=rho_cl,
            w_orient=w_orient, m_orient=m_orient, n_orient=n_orient,
            gds=gds,
        )

    if len(all_data) < 2:
        print("\nNot enough systems loaded for comparison plots.")
        return

    # Overlay comparison plots
    comp_dir = os.path.join(args.outdir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    pc = lambda n: os.path.join(comp_dir, n)

    comparison_line(all_data, "rho_meoh",
                    "Methanol Density – All Concentrations",
                    "atoms/Å³", pc("comp_methanol_density"))
    comparison_line(all_data, "rho_nacl",
                    "NaCl Density – All Concentrations",
                    "atoms/Å³", pc("comp_nacl_density"))
    comparison_na_cl(all_data, pc("comp_na_vs_cl"))
    comparison_line(all_data, "w_orient",
                    "Water Orientation – All Concentrations",
                    r"$\langle\cos\theta\rangle$", pc("comp_water_orient"),
                    ylim=(-1, 1), zero_line=True)
    comparison_line(all_data, "m_orient",
                    "Methanol Orientation – All Concentrations",
                    r"$\langle\cos\theta\rangle$", pc("comp_methanol_orient"),
                    ylim=(-1, 1), zero_line=True)
    comparison_line(all_data, "n_orient",
                    "NaCl Orientation – All Concentrations",
                    r"$\langle\cos\theta\rangle$", pc("comp_nacl_orient"),
                    ylim=(-1, 1), zero_line=True, nan_ok=True)
    summary_panel(all_data, pc("comp_summary"))

    # Master CSV
    with open(os.path.join(args.outdir, "summary.csv"), "w") as fh:
        fh.write("sim,z_A,rho_water,rho_methanol,rho_nacl,"
                 "rho_na,rho_cl,w_orient,m_orient,n_orient\n")
        for name, d in all_data.items():
            z  = d["z"]
            no = np.where(np.isnan(d["n_orient"]), 0.0, d["n_orient"])
            for i in range(len(z)):
                fh.write(
                    f"{name},{z[i]:.2f},{d['rho_water'][i]:.6e},"
                    f"{d['rho_meoh'][i]:.6e},{d['rho_nacl'][i]:.6e},"
                    f"{d['rho_na'][i]:.6e},{d['rho_cl'][i]:.6e},"
                    f"{d['w_orient'][i]:.6f},{d['m_orient'][i]:.6f},{no[i]:.6f}\n"
                )

    print(f"\n  Done. All outputs in: {args.outdir}/")


if __name__ == "__main__":
    main()
