#!/usr/bin/env python3
"""
generate_packmol_input.py
=========================
Generate a PACKMOL input file (water_slab.inp) for the water-NaCl slab system.

System geometry:
  Box    : 60 × 60 × 200 Å, periodic in x/y, slab in z
  Slab   : z = 30–170 Å  (water + ions)
  Vapour : z = 0–30 Å (bottom) and z = 170–200 Å (top) — vacuum

Both ions (Na⁺ and Cl⁻) are placed inside the water slab where they belong
physically (they are strongly solvated and do not exist in vapour phase).

Force fields used downstream:
  Water : SPC/Fw  (Wu et al., J. Chem. Phys. 124, 024503, 2006)
  Ions  : Joung-Cheatham (J. Phys. Chem. B 112, 9020, 2008) for SPC/E,
           cross-validated for SPC/Fw slab geometry

Usage:
  python3 generate_packmol_input.py <n_nacl_pairs> [output_file]
  python3 generate_packmol_input.py 10          # → 10 Na+ and 10 Cl-
  python3 generate_packmol_input.py 50          # → 50 ion pairs
"""
import sys


def generate(n_nacl: int, output_file: str = "water_slab.inp") -> None:
    lx, ly       = 60.0, 60.0
    lz           = 200.0
    slab_lo      = 30.0
    slab_hi      = 170.0

    # Water count: fills 60×60×140 Å at ~1 g/cm³
    # 16632 gives slightly more space for ions vs the full 16800 pure-water count
    n_water = 16632

    with open(output_file, "w") as f:
        f.write("tolerance 2.0\n")
        f.write("filetype pdb\n")
        f.write("output water_slab.pdb\n")
        f.write("seed 12345\n\n")

        # Water slab
        f.write("# Water slab\n")
        f.write("structure water.pdb\n")
        f.write(f"  number {n_water}\n")
        f.write(f"  inside box 0.0 0.0 {slab_lo} {lx} {ly} {slab_hi}\n")
        f.write("end structure\n\n")

        # Na+ ions — placed inside water slab
        if n_nacl > 0:
            f.write("# Sodium ions (inside water slab)\n")
            f.write("structure na.pdb\n")
            f.write(f"  number {n_nacl}\n")
            f.write(f"  inside box 0.0 0.0 {slab_lo} {lx} {ly} {slab_hi}\n")
            f.write("end structure\n\n")

            # Cl- ions — placed inside water slab
            f.write("# Chloride ions (inside water slab)\n")
            f.write("structure cl.pdb\n")
            f.write(f"  number {n_nacl}\n")
            f.write(f"  inside box 0.0 0.0 {slab_lo} {lx} {ly} {slab_hi}\n")
            f.write("end structure\n\n")

    print(f"Generated {output_file}:")
    print(f"  Water     : {n_water}")
    print(f"  Na+ ions  : {n_nacl}")
    print(f"  Cl- ions  : {n_nacl}")
    print(f"  Total atoms: {n_water * 3 + n_nacl * 2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_packmol_input.py <n_nacl_pairs> [output_file]")
        sys.exit(1)
    n = int(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "water_slab.inp"
    generate(n, out)
