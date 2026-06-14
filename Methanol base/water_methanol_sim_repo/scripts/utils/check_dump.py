#!/usr/bin/env python3
"""
check_dump.py
=============
Quick diagnostic for the water-methanol trajectory file.

Checks:
  - Column layout (flags missing mol, type, x, y, z)
  - Atom-type counts per frame
  - Molecule grouping via mol-ID (flags positional-indexing failures)
  - Water and methanol z-distributions (flags atoms outside expected regions)

Usage:
  python3 check_dump.py [--dump traj.lammpstrj] [--ncheck 2]
"""
import os, sys, argparse
import numpy as np

T_O_WAT = 1; T_H_WAT = 2
T_C_MET = 3; T_O_MET = 4; T_H_MET = 5; T_H_OH = 6

LABELS = {
    1: "O_water",  2: "H_water",
    3: "C_methanol", 4: "O_methanol",
    5: "H_methyl",   6: "H_hydroxyl",
}

DUMP_CANDIDATES = ["traj.lammpstrj", "traj_combined.lammpstrj"]


def read_n_frames(filename, n):
    frames = []
    with open(filename) as fh:
        while len(frames) < n:
            line = fh.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue
            step   = fh.readline().strip()
            fh.readline()
            natoms = int(fh.readline().strip())
            fh.readline(); fh.readline(); fh.readline()
            hdr    = fh.readline().strip()
            cols   = hdr.split()[2:]
            rows   = [fh.readline().split() for _ in range(natoms)]
            frames.append((step, cols, rows))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump",   default=None)
    parser.add_argument("--ncheck", default=2, type=int)
    args = parser.parse_args()

    dump = args.dump
    if dump is None:
        for c in DUMP_CANDIDATES:
            if os.path.exists(c):
                dump = c; break
    if dump is None or not os.path.exists(dump):
        sys.exit("ERROR: No trajectory file found. Use --dump <path>")

    print(f"\n{'='*60}")
    print(f"  File : {dump}  ({os.path.getsize(dump)/1e9:.2f} GB)")
    print(f"{'='*60}\n")

    frames = read_n_frames(dump, args.ncheck)
    if not frames:
        sys.exit("ERROR: No frames readable.")

    for fi, (step, cols, rows) in enumerate(frames):
        print(f"  ── Frame {fi+1}  (timestep {step}) ─────────────────────")
        print(f"     Columns : {cols}")

        required = ["id", "type", "x", "y", "z"]
        missing  = [c for c in required if c not in cols]
        status   = "✓" if not missing else "✗"
        print(f"  {status}  Required columns: "
              + ("all present" if not missing else f"MISSING {missing}"))

        has_mol = "mol" in cols
        print(f"  {'✓' if has_mol else '✗'}  mol column: "
              + ("PRESENT" if has_mol else
                 "ABSENT – analysis scripts require this column"))

        cmap     = {c: i for i, c in enumerate(cols)}
        idx_type = cmap.get("type", 1)
        idx_mol  = cmap.get("mol", -1)
        idx_z    = cmap.get("z", 4)

        atypes   = [int(r[idx_type]) for r in rows]
        zvals    = [float(r[idx_z])  for r in rows]
        tc       = {}
        for t in atypes: tc[t] = tc.get(t, 0) + 1

        print("\n     Atom-type counts:")
        for t in sorted(tc):
            print(f"       {t:2d}  ({LABELS.get(t,'?'):12s}) : {tc[t]}")

        n_owater = tc.get(T_O_WAT, 0)
        n_cmet   = tc.get(T_C_MET, 0)
        print(f"\n     Inferred: water={n_owater},  methanol={n_cmet}")

        if has_mol:
            mol_ids = [int(r[idx_mol]) for r in rows]
            mol_type_map = {}
            for mid, at in zip(mol_ids, atypes):
                mol_type_map.setdefault(mid, set()).add(at)
            n_wat_mols  = sum(1 for s in mol_type_map.values()
                              if s == {T_O_WAT, T_H_WAT})
            n_met_mols  = sum(1 for s in mol_type_map.values()
                              if T_C_MET in s and T_O_WAT not in s)
            print(f"     Mol-ID groups: water={n_wat_mols},  methanol={n_met_mols}")
            if n_met_mols == 0 and n_cmet > 0:
                print("  ✗  Methanol mol grouping FAILED (check dump mol column)")

        # z-distribution check
        slab_lo, slab_hi = 25.0, 175.0
        wat_z   = [zvals[i] for i, t in enumerate(atypes) if t == T_O_WAT]
        in_slab = sum(1 for z in wat_z if slab_lo < z < slab_hi)
        print(f"\n     Water O:  {len(wat_z)} atoms,  {in_slab} in slab "
              f"({slab_lo}-{slab_hi} A)")

        met_z = [zvals[i] for i, t in enumerate(atypes) if t == T_C_MET]
        if met_z:
            print(f"     Methanol C:  {len(met_z)} atoms,  "
                  f"z range {min(met_z):.1f}-{max(met_z):.1f} A")
        print()

    print("=" * 60)
    print("  TIP: if mol column is absent, add 'mol' to the dump line in")
    print("  in.lammps:  dump TRAJ all custom 1000 traj.lammpstrj id mol type x y z vx vy vz")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
