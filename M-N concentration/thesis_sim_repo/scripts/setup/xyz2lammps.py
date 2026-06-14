#!/usr/bin/env python3
"""
xyz2lammps.py
=============
Convert a PACKMOL-generated system.xyz into a LAMMPS full-style data file
(system.data) with complete topology: bonds, angles, charges, and masses.

Must be run in the same directory as:
  - system.xyz    (output from PACKMOL)
  - pack.inp      (input used to generate system.xyz; used to count molecules)

Atom types and force-field mapping:
  Type 1 – O  (water,   SPC/Fw)   charge: -0.82 e,  mass: 15.9994
  Type 2 – H  (water,   SPC/Fw)   charge: +0.41 e,  mass:  1.00794
  Type 3 – C  (methanol, OPLS-AA) charge: -0.18 e,  mass: 12.0110
  Type 4 – O  (methanol, OPLS-AA) charge: -0.683 e, mass: 15.9994
  Type 5 – H  (methyl H, OPLS-AA) charge: +0.145 e, mass:  1.00794
  Type 6 – H  (hydroxyl H, OPLS-AA) charge: +0.418 e, mass:  1.00794
  Type 7 – Na⁺ (Joung-Cheatham)  charge: +1.0 e,   mass: 22.9898
  Type 8 – Cl⁻ (Joung-Cheatham)  charge: -1.0 e,   mass: 35.4530

Bond types:
  1 – O–H  (water)
  2 – C–O  (methanol)
  3 – O–H  (methanol hydroxyl)
  4 – C–H  (methanol methyl)

Angle types:
  1 – H–O–H (water)
  2 – C–O–H (methanol)
  3 – H–C–H (methanol)
  4 – H–C–O (methanol)
"""
import numpy as np

# ─── Read system.xyz ────────────────────────────────────────────────────────
with open('system.xyz', 'r') as f:
    lines = f.readlines()

natoms = int(lines[0].strip())
print(f"Total atoms: {natoms}")

lx, ly, lz = 60.0, 60.0, 200.0

# ─── Parse molecule counts from pack.inp ────────────────────────────────────
n_water = 16800
n_methanol_top = n_methanol_bot = 0
n_na_top = n_na_bot = 0
n_cl_top = n_cl_bot = 0

with open('pack.inp', 'r') as f:
    lines_pack = f.readlines()
    for i, line in enumerate(lines_pack):
        if 'structure methanol.xyz' in line:
            num = int(lines_pack[i+1].split()[1])
            if '0. 0. 0.0' in lines_pack[i+2]:
                n_methanol_bot = num
            else:
                n_methanol_top = num
        elif 'structure na.xyz' in line:
            num = int(lines_pack[i+1].split()[1])
            if '0. 0. 0.0' in lines_pack[i+2]:
                n_na_bot = num
            else:
                n_na_top = num
        elif 'structure cl.xyz' in line:
            num = int(lines_pack[i+1].split()[1])
            if '0. 0. 0.0' in lines_pack[i+2]:
                n_cl_bot = num
            else:
                n_cl_top = num

n_methanol = n_methanol_top + n_methanol_bot
n_nacl     = n_na_top + n_na_bot
print(f"Detected: water={n_water}, methanol={n_methanol}, NaCl={n_nacl}")

# ─── Build topology ──────────────────────────────────────────────────────────
atom_id = bond_id = angle_id = 1
mol_id  = 1
atoms   = []
bonds   = []
angles  = []

# Read coordinates from XYZ (offset +2 for header lines)
offset_water = 2

# ── Water molecules ──────────────────────────────────────────────────────────
for w in range(n_water):
    idx = offset_water + w * 3
    o_line  = lines[idx].split()
    h1_line = lines[idx+1].split()
    h2_line = lines[idx+2].split()

    o_id, h1_id, h2_id = atom_id, atom_id+1, atom_id+2

    atoms.append(f"{o_id}  {mol_id} 1 -0.82  {o_line[1]}  {o_line[2]}  {o_line[3]}")
    atoms.append(f"{h1_id} {mol_id} 2  0.41  {h1_line[1]} {h1_line[2]} {h1_line[3]}")
    atoms.append(f"{h2_id} {mol_id} 2  0.41  {h2_line[1]} {h2_line[2]} {h2_line[3]}")

    bonds.append(f"{bond_id} 1 {o_id} {h1_id}");  bond_id += 1
    bonds.append(f"{bond_id} 1 {o_id} {h2_id}");  bond_id += 1
    angles.append(f"{angle_id} 1 {h1_id} {o_id} {h2_id}"); angle_id += 1

    atom_id += 3
    mol_id  += 1

# ── Methanol molecules ───────────────────────────────────────────────────────
offset_methanol = offset_water + n_water * 3

for m in range(n_methanol):
    idx = offset_methanol + m * 6
    c_line    = lines[idx].split()
    o_line    = lines[idx+1].split()
    h_oh_line = lines[idx+2].split()
    h1_line   = lines[idx+3].split()
    h2_line   = lines[idx+4].split()
    h3_line   = lines[idx+5].split()

    c_id   = atom_id;     o_id   = atom_id+1; h_oh_id = atom_id+2
    h1_id  = atom_id+3;   h2_id  = atom_id+4; h3_id   = atom_id+5

    atoms.append(f"{c_id}    {mol_id} 3 -0.18   {c_line[1]}    {c_line[2]}    {c_line[3]}")
    atoms.append(f"{o_id}    {mol_id} 4 -0.683  {o_line[1]}    {o_line[2]}    {o_line[3]}")
    atoms.append(f"{h_oh_id} {mol_id} 6  0.418  {h_oh_line[1]} {h_oh_line[2]} {h_oh_line[3]}")
    atoms.append(f"{h1_id}   {mol_id} 5  0.145  {h1_line[1]}   {h1_line[2]}   {h1_line[3]}")
    atoms.append(f"{h2_id}   {mol_id} 5  0.145  {h2_line[1]}   {h2_line[2]}   {h2_line[3]}")
    atoms.append(f"{h3_id}   {mol_id} 5  0.145  {h3_line[1]}   {h3_line[2]}   {h3_line[3]}")

    bonds.append(f"{bond_id} 2 {c_id} {o_id}");     bond_id += 1
    bonds.append(f"{bond_id} 3 {o_id} {h_oh_id}");  bond_id += 1
    for h_id in [h1_id, h2_id, h3_id]:
        bonds.append(f"{bond_id} 4 {c_id} {h_id}"); bond_id += 1

    angles.append(f"{angle_id} 2 {c_id} {o_id} {h_oh_id}"); angle_id += 1
    for hA, hB in [(h1_id, h2_id), (h1_id, h3_id), (h2_id, h3_id)]:
        angles.append(f"{angle_id} 3 {hA} {c_id} {hB}"); angle_id += 1
    for h_id in [h1_id, h2_id, h3_id]:
        angles.append(f"{angle_id} 4 {h_id} {c_id} {o_id}"); angle_id += 1

    atom_id += 6
    mol_id  += 1

# ── Na⁺ ions ─────────────────────────────────────────────────────────────────
offset_na = offset_methanol + n_methanol * 6
for i in range(n_nacl):
    na_line = lines[offset_na + i].split()
    atoms.append(f"{atom_id} {mol_id} 7  1.0  {na_line[1]} {na_line[2]} {na_line[3]}")
    atom_id += 1
    mol_id  += 1

# ── Cl⁻ ions ─────────────────────────────────────────────────────────────────
offset_cl = offset_na + n_nacl
for i in range(n_nacl):
    cl_line = lines[offset_cl + i].split()
    atoms.append(f"{atom_id} {mol_id} 8 -1.0  {cl_line[1]} {cl_line[2]} {cl_line[3]}")
    atom_id += 1
    mol_id  += 1

print(f"Total molecules: {mol_id - 1}")
print(f"Bonds: {len(bonds)},  Angles: {len(angles)}")

# ─── Write LAMMPS data file ──────────────────────────────────────────────────
with open('system.data', 'w') as f:
    f.write("LAMMPS data file – water/methanol/NaCl liquid–vapour interface\n\n")
    f.write(f"{natoms} atoms\n")
    f.write(f"{len(bonds)} bonds\n")
    f.write(f"{len(angles)} angles\n")
    f.write("0 dihedrals\n0 impropers\n\n")
    f.write("8 atom types\n4 bond types\n4 angle types\n\n")
    f.write(f"0.0 {lx} xlo xhi\n")
    f.write(f"0.0 {ly} ylo yhi\n")
    f.write(f"0.0 {lz} zlo zhi\n\n")
    f.write("Masses\n\n")
    f.write("1 15.9994  # O  (water, SPC/Fw)\n")
    f.write("2  1.00794 # H  (water, SPC/Fw)\n")
    f.write("3 12.0110  # C  (methanol, OPLS-AA)\n")
    f.write("4 15.9994  # O  (methanol, OPLS-AA)\n")
    f.write("5  1.00794 # H  (methyl, OPLS-AA)\n")
    f.write("6  1.00794 # H  (hydroxyl, OPLS-AA)\n")
    f.write("7 22.9898  # Na+ (Joung-Cheatham)\n")
    f.write("8 35.4530  # Cl- (Joung-Cheatham)\n\n")
    f.write("Atoms\n\n")
    for atom in atoms:
        f.write(atom + "\n")
    f.write("\nBonds\n\n")
    for bond in bonds:
        f.write(bond + "\n")
    f.write("\nAngles\n\n")
    for angle in angles:
        f.write(angle + "\n")

print("LAMMPS data file written: system.data")
