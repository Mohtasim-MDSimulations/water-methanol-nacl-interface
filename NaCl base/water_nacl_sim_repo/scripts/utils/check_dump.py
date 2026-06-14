#!/usr/bin/env python3
"""
check_dump.py
=============
Quick diagnostic for the water-NaCl trajectory file.

Checks:
  - Column layout (required: id, mol, type, x, y, z)
  - Atom-type counts
  - Mol-ID grouping sanity (water = exactly 3 atoms per mol)
  - Ion z-distribution (warns if ions are in vapour region)

Usage:
  python3 check_dump.py [--dump production.lammpstrj] [--ncheck 2]
"""
import os, sys, argparse

T_O_WAT = 1; T_H_WAT = 2; T_NA = 3; T_CL = 4

LABELS = {1: "O_water", 2: "H_water", 3: "Na+", 4: "Cl-"}
DUMP_CANDIDATES = ["production.lammpstrj", "equilibration.lammpstrj"]


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
        sys.exit("ERROR: No readable frames.")

    for fi, (step, cols, rows) in enumerate(frames):
        print(f"  ── Frame {fi+1}  (timestep {step}) ─────────────────────")
        print(f"     Columns : {cols}")

        required = ["id", "mol", "type", "x", "y", "z"]
        missing  = [c for c in required if c not in cols]
        print(f"  {'✓' if not missing else '✗'}  Required columns: "
              + ("all present" if not missing else f"MISSING {missing}"))

        cmap     = {c: i for i, c in enumerate(cols)}
        idx_type = cmap.get("type", 2)
        idx_mol  = cmap.get("mol",  1)
        idx_z    = cmap.get("z",    5)

        atypes = [int(r[idx_type]) for r in rows]
        mols   = [int(r[idx_mol])  for r in rows]
        zvals  = [float(r[idx_z])  for r in rows]

        tc = {}
        for t in atypes: tc[t] = tc.get(t, 0) + 1
        print("\n     Atom-type counts:")
        for t in sorted(tc):
            print(f"       {t}  ({LABELS.get(t,'?'):10s}) : {tc[t]}")

        # Mol grouping
        mol_map = {}
        for mol, at in zip(mols, atypes):
            mol_map.setdefault(mol, set()).add(at)

        n_water = sum(1 for s in mol_map.values() if s == {T_O_WAT, T_H_WAT})
        n_na    = sum(1 for s in mol_map.values() if s == {T_NA})
        n_cl    = sum(1 for s in mol_map.values() if s == {T_CL})
        print(f"\n     Mol groups: water={n_water},  Na+={n_na},  Cl-={n_cl}")

        # Ion z-range
        SLAB_LO, SLAB_HI = 25.0, 175.0
        for ion_type, label in [(T_NA, "Na+"), (T_CL, "Cl-")]:
            ion_z = [zvals[i] for i, t in enumerate(atypes) if t == ion_type]
            if ion_z:
                in_slab = sum(1 for z in ion_z if SLAB_LO < z < SLAB_HI)
                in_vap  = len(ion_z) - in_slab
                status  = "✓" if in_vap == 0 else "✗"
                print(f"  {status}  {label}: z={min(ion_z):.1f}–{max(ion_z):.1f} Å  "
                      f"(slab:{in_slab}, vapour:{in_vap})")
                if in_vap > 0:
                    print(f"     WARNING: {in_vap} {label} ions outside slab — "
                          f"may indicate equilibration issue")
        print()

    print("=" * 60)
    print("  TIP: timestep should start near 0 in production dump")
    print("  (production.in uses reset_timestep 0)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
