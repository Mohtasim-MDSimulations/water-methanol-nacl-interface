#!/usr/bin/env python3
"""
check_dump.py
=============
Quick diagnostic tool for LAMMPS trajectory (dump) files.

Reads the first N frames and reports:
  - Column layout (checks for required id, type, x, y, z; flags missing 'mol')
  - Atom-type count per frame
  - Inferred molecule counts (water, methanol, Na⁺, Cl⁻)
  - Ion z-positions (flags any ions that escaped into the vapour region)
  - Methanol mol-ID grouping (flags silent-zero bug from missing mol column)

Usage:
  python3 check_dump.py [--dump traj.lammpstrj] [--ncheck 2]

Run this before analysis to catch topology or I/O issues early.
"""
import os, sys, argparse
import numpy as np

# ── Atom-type constants ───────────────────────────────────────────────────────
T_O_WAT = 1; T_H_WAT = 2
T_C_MET = 3; T_O_MET = 4; T_H_MET = 5; T_H_OH = 6
T_NA    = 7; T_CL    = 8
METHANOL_TYPES = {T_C_MET, T_O_MET, T_H_MET, T_H_OH}

DUMP_CANDIDATES = [
    "traj.lammpstrj",
    "traj_combined.lammpstrj",
    "traj_new.lammpstrj",
    "traj_continued.lammpstrj",
]

LABELS = {
    1: "O_water",    2: "H_water",
    3: "C_methanol", 4: "O_methanol",
    5: "H_methyl",   6: "H_hydroxyl",
    7: "Na+",        8: "Cl-",
}

# ── Frame reader ──────────────────────────────────────────────────────────────
def read_n_frames(filename, n):
    frames = []
    with open(filename, "r") as fh:
        while len(frames) < n:
            line = fh.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue
            step    = fh.readline().strip()
            fh.readline()
            natoms  = int(fh.readline().strip())
            fh.readline(); fh.readline(); fh.readline()
            atoms_hdr = fh.readline().strip()
            cols  = atoms_hdr.replace("ITEM: ATOMS", "").split()
            rows  = [fh.readline().split() for _ in range(natoms)]
            frames.append((step, cols, rows))
    return frames


# ── Main diagnostic ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Diagnostic check for LAMMPS trajectory files")
    parser.add_argument("--dump",   default=None,  help="Path to dump file")
    parser.add_argument("--ncheck", default=2, type=int, help="Number of frames to check (default: 2)")
    args = parser.parse_args()

    # Locate dump file
    dump = args.dump
    if dump is None:
        for c in DUMP_CANDIDATES:
            if os.path.exists(c):
                dump = c
                break
    if dump is None or not os.path.exists(dump):
        sys.exit("ERROR: No trajectory file found. Specify with --dump <path>")

    size_gb = os.path.getsize(dump) / 1e9
    print(f"\n{'='*62}")
    print(f"  Dump file : {dump}")
    print(f"  Size      : {size_gb:.2f} GB")
    print(f"{'='*62}\n")

    frames = read_n_frames(dump, args.ncheck)
    if not frames:
        sys.exit("ERROR: Could not read any frames from dump file.")
    print(f"  Frames read: {len(frames)}\n")

    for fi, (step, cols, rows) in enumerate(frames):
        print(f"  ── Frame {fi+1}  (timestep {step}) ──────────────────────────")
        print(f"     Columns : {cols}")

        # Required column check
        required = ["id", "type", "x", "y", "z"]
        missing  = [c for c in required if c not in cols]
        if missing:
            print(f"  ✗  MISSING required columns: {missing}")
        else:
            print(f"  ✓  All required columns present")

        has_mol = "mol" in cols
        print(f"  {'✓' if has_mol else '✗'}  'mol' column: {'PRESENT' if has_mol else 'ABSENT'}")
        if not has_mol:
            print("     → analysis scripts will load mol IDs from system.data")

        cmap = {c: i for i, c in enumerate(cols)}
        idx_type = cmap.get("type", 1)
        idx_mol  = cmap.get("mol", -1)
        idx_z    = cmap.get("z", 6)

        atypes = [int(r[idx_type]) for r in rows]
        zvals  = [float(r[idx_z])  for r in rows]

        # Atom-type counts
        type_counts = {}
        for t in atypes:
            type_counts[t] = type_counts.get(t, 0) + 1
        print("\n     Atom-type counts:")
        for t in sorted(type_counts):
            print(f"       {t:2d}  ({LABELS.get(t, '?'):12s}) : {type_counts[t]}")

        n_owater = type_counts.get(T_O_WAT, 0)
        n_cmet   = type_counts.get(T_C_MET, 0)
        n_na     = type_counts.get(T_NA,    0)
        n_cl     = type_counts.get(T_CL,    0)
        print(f"\n     Inferred molecules – "
              f"Water: {n_owater}  Methanol: {n_cmet}  Na⁺: {n_na}  Cl⁻: {n_cl}")

        # Mol-ID grouping sanity check (if mol column present)
        if has_mol:
            mol_ids = [int(r[idx_mol]) for r in rows]
            mol_type_map = {}
            for mid, at in zip(mol_ids, atypes):
                mol_type_map.setdefault(mid, set()).add(at)
            n_water_mols = sum(1 for s in mol_type_map.values() if s == {T_O_WAT, T_H_WAT})
            n_meoh_mols  = sum(1 for s in mol_type_map.values()
                               if METHANOL_TYPES.issubset(s) and T_O_WAT not in s)
            print(f"     Mol-ID groups – water: {n_water_mols}  methanol: {n_meoh_mols}")
            if n_meoh_mols == 0 and n_cmet > 0:
                print("  ✗  Methanol mol-ID grouping BROKEN (silent-zero bug)")

        # Ion z-range / vapour escape check
        na_z = [zvals[i] for i, t in enumerate(atypes) if t == T_NA]
        cl_z = [zvals[i] for i, t in enumerate(atypes) if t == T_CL]
        for ion_label, ion_z in [("Na⁺", na_z), ("Cl⁻", cl_z)]:
            if ion_z:
                in_slab = sum(1 for z in ion_z if 28 < z < 172)
                in_vap  = len(ion_z) - in_slab
                print(f"     {ion_label} z-range: {min(ion_z):.1f}–{max(ion_z):.1f} Å  "
                      f"(slab: {in_slab}, vapour: {in_vap})")
                if in_vap > 0:
                    print(f"  ✗  {ion_label} IONS IN VAPOUR – density profile will show artefact")

        print()

    print("=" * 62)
    print("  TIP: If 'mol' column is ABSENT, all analysis scripts")
    print("       automatically repair it from system.data.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
