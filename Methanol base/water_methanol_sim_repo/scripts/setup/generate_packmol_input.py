#!/usr/bin/env python3
"""
generate_packmol_input.py
=========================
Generate a PACKMOL input file (pack.inp) for a water–methanol slab simulation.

System geometry:
  Box    : 60 × 60 × 200 Å, fully periodic
  Slab   : z = 30–170 Å  (water, ~1 g/cm³ at 16,800 molecules)
  Vapour : z = 0–30 Å (bottom) and z = 170–200 Å (top)

Methanol placement:
  Molecules are split EQUALLY between the two vapour/interface regions so
  that both interfaces start with the same composition.
  (BUG FIX: original pack.inp placed all methanol in the top region only,
  biasing adsorption asymmetrically from the start.)

Force fields used downstream:
  Water    – SPC/Fw  (Wu et al., J. Chem. Phys. 124, 024503, 2006)
  Methanol – OPLS-AA (Jorgensen et al.)

Usage:
  python3 generate_packmol_input.py <n_methanol>
  python3 generate_packmol_input.py 10    # → 5 methanol top, 5 bottom
  python3 generate_packmol_input.py 20    # → 10 top, 10 bottom
"""
import sys


def generate_packmol_input(n_methanol: int, output_file: str = "pack.inp") -> None:
    lx, ly, lz          = 60.0, 60.0, 200.0
    water_z_lo          = 30.0
    water_z_hi          = 170.0
    vapor_bot_lo        = 0.0
    vapor_bot_hi        = 30.0
    vapor_top_lo        = 170.0
    vapor_top_hi        = 200.0

    n_water             = 16800

    # Split methanol equally between top and bottom interfaces
    n_methanol_top      = n_methanol // 2 + (n_methanol % 2)   # ceiling half
    n_methanol_bot      = n_methanol // 2                        # floor half

    with open(output_file, "w") as f:
        f.write("tolerance 2.0\n")
        f.write("filetype xyz\n")
        f.write("output system.xyz\n\n")

        # Water slab
        f.write("# Water slab  (~140 Å thick, ~1 g/cm³)\n")
        f.write("structure water.xyz\n")
        f.write(f"  number {n_water}\n")
        f.write(f"  inside box 0. 0. {water_z_lo} {lx} {ly} {water_z_hi}\n")
        f.write("end structure\n\n")

        # Methanol – bottom vapour region
        if n_methanol_bot > 0:
            f.write("# Methanol – bottom vapour / interface region\n")
            f.write("structure methanol.xyz\n")
            f.write(f"  number {n_methanol_bot}\n")
            f.write(f"  inside box 0. 0. {vapor_bot_lo} {lx} {ly} {vapor_bot_hi}\n")
            f.write("end structure\n\n")

        # Methanol – top vapour region
        if n_methanol_top > 0:
            f.write("# Methanol – top vapour / interface region\n")
            f.write("structure methanol.xyz\n")
            f.write(f"  number {n_methanol_top}\n")
            f.write(f"  inside box 0. 0. {vapor_top_lo} {lx} {ly} {vapor_top_hi}\n")
            f.write("end structure\n\n")

    print(f"Generated {output_file}:  water={n_water},  "
          f"methanol={n_methanol} ({n_methanol_bot} bottom + {n_methanol_top} top)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_packmol_input.py <n_methanol>")
        sys.exit(1)
    generate_packmol_input(int(sys.argv[1]))
