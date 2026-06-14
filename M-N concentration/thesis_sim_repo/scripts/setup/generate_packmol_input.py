#!/usr/bin/env python3
"""
generate_packmol_input.py
=========================
Generate a PACKMOL input file for a water-methanol-NaCl slab simulation.

System geometry:
  Box: 60 × 60 × 200 Å (periodic in all directions)
  Water slab: z = 30–170 Å  (~140 Å thick)
  Vapor regions: z = 0–30 Å (bottom) and z = 170–200 Å (top)
  Ions: placed exclusively inside the water slab
  Methanol: split equally between the two vapor/interface regions

Force fields:
  Water    – SPC/Fw (Wu et al., J. Chem. Phys. 124, 024503, 2006)
  Methanol – OPLS-AA (Jorgensen et al.)
  Na⁺/Cl⁻ – Joung-Cheatham (J. Phys. Chem. B 112, 9020, 2008)

Usage:
  python3 generate_packmol_input.py <n_methanol> <n_nacl>

  Examples:
    python3 generate_packmol_input.py 0   0    # pure water baseline
    python3 generate_packmol_input.py 0  10    # pure NaCl (N10)
    python3 generate_packmol_input.py 10  0    # pure methanol (M10)
    python3 generate_packmol_input.py 10 10    # mixed M10N10
    python3 generate_packmol_input.py 100 100  # mixed M100N100
"""
import sys


def generate_packmol_input(n_methanol, n_nacl, output_file="pack.inp"):
    lx, ly, lz = 60.0, 60.0, 200.0
    water_min_z, water_max_z = 30.0, 170.0
    vapor_min_z_bot, vapor_max_z_bot = 0.0, 30.0
    vapor_min_z_top, vapor_max_z_top = 170.0, 200.0

    # Fixed water count: 16,800 molecules fill the 60×60×140 Å slab at ~1 g/cm³
    n_water = 16800

    # Split methanol molecules evenly across top/bottom vapor–interface regions
    n_methanol_top = n_methanol // 2 + (n_methanol % 2)
    n_methanol_bot = n_methanol // 2

    # All ions go into the water slab (split top/bottom half of slab)
    n_nacl_top = n_nacl // 2 + (n_nacl % 2)
    n_nacl_bot = n_nacl // 2

    with open(output_file, 'w') as f:
        f.write("tolerance 2.0\n")
        f.write("filetype xyz\n")
        f.write("output system.xyz\n\n")

        # Water slab
        f.write(f"structure water.xyz\n")
        f.write(f"  number {n_water}\n")
        f.write(f"  inside box 0. 0. {water_min_z} {lx} {ly} {water_max_z}\n")
        f.write(f"end structure\n\n")

        # Methanol – bottom vapor region
        if n_methanol_bot > 0:
            f.write(f"structure methanol.xyz\n")
            f.write(f"  number {n_methanol_bot}\n")
            f.write(f"  inside box 0. 0. {vapor_min_z_bot} {lx} {ly} {vapor_max_z_bot}\n")
            f.write(f"end structure\n\n")

        # Methanol – top vapor region
        if n_methanol_top > 0:
            f.write(f"structure methanol.xyz\n")
            f.write(f"  number {n_methanol_top}\n")
            f.write(f"  inside box 0. 0. {vapor_min_z_top} {lx} {ly} {vapor_max_z_top}\n")
            f.write(f"end structure\n\n")

        # Na⁺ ions – placed inside water slab (bottom half)
        if n_nacl_bot > 0:
            f.write(f"structure na.xyz\n")
            f.write(f"  number {n_nacl_bot}\n")
            f.write(f"  inside box 0. 0. {water_min_z} {lx} {ly} {water_max_z}\n")
            f.write(f"end structure\n\n")
            f.write(f"structure cl.xyz\n")
            f.write(f"  number {n_nacl_bot}\n")
            f.write(f"  inside box 0. 0. {water_min_z} {lx} {ly} {water_max_z}\n")
            f.write(f"end structure\n\n")

        # Na⁺/Cl⁻ ions – placed inside water slab (top half)
        if n_nacl_top > 0:
            f.write(f"structure na.xyz\n")
            f.write(f"  number {n_nacl_top}\n")
            f.write(f"  inside box 0. 0. {water_min_z} {lx} {ly} {water_max_z}\n")
            f.write(f"end structure\n\n")
            f.write(f"structure cl.xyz\n")
            f.write(f"  number {n_nacl_top}\n")
            f.write(f"  inside box 0. 0. {water_min_z} {lx} {ly} {water_max_z}\n")
            f.write(f"end structure\n\n")

    print(f"Generated {output_file}:  methanol={n_methanol}, NaCl={n_nacl}, water={n_water}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_packmol_input.py <n_methanol> <n_nacl>")
        sys.exit(1)
    generate_packmol_input(int(sys.argv[1]), int(sys.argv[2]))
