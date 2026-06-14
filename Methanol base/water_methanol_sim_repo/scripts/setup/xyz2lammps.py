#!/usr/bin/env python3
"""
xyz2lammps.py
=============
Convert a PACKMOL-generated system.xyz into a LAMMPS full-style data file
(system.data) with complete topology: bonds, angles, charges, and masses.

Must be run in the same directory as:
  - system.xyz   (PACKMOL output)
  - pack.inp     (PACKMOL input; parsed to determine molecule counts)

BUG FIX vs original:
  The original script hardcoded nwater=16800 and nmethanol=10. If pack.inp
  is changed (different methanol count, different run), the converter
  silently produced the wrong atom ordering and topology.
  This version parses pack.inp dynamically to extract the correct counts.

Atom types:
  1 – O  (water,    SPC/Fw)     charge: −0.82 e
  2 – H  (water,    SPC/Fw)     charge: +0.41 e
  3 – C  (methanol, OPLS-AA)    charge: −0.18 e
  4 – O  (methanol, OPLS-AA)    charge: −0.683 e
  5 – H  (methyl H, OPLS-AA)    charge: +0.145 e
  6 – H  (hydroxyl, OPLS-AA)    charge: +0.418 e

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

# ── Parse pack.inp to get molecule counts ─────────────────────────────────────
def parse_pack_inp(filename="pack.inp"):
    """
    Reads pack.inp and returns (n_water, n_methanol_bot, n_methanol_top).
    Distinguishes methanol placement by which z-region is used.
    """
    n_water        = 0
    n_methanol_bot = 0
    n_methanol_top = 0

    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("structure water.xyz"):
            # Next non-empty line should be "number N"
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            num_line = lines[j].strip().split()
            if num_line[0] == "number":
                n_water = int(num_line[1])

        elif line.startswith("structure methanol.xyz"):
            j = i + 1
            # Read 'number' line
            while j < len(lines) and not lines[j].strip():
                j += 1
            num_line = lines[j].strip().split()
            count = int(num_line[1]) if num_line[0] == "number" else 0
            j += 1
            # Read 'inside box ...' line
            while j < len(lines) and not lines[j].strip():
                j += 1
            box_line = lines[j].strip()
            if "inside box" in box_line:
                # "inside box xlo ylo zlo xhi yhi zhi"
                parts = box_line.split()
                # parts: ['inside', 'box', xlo, ylo, zlo, xhi, yhi, zhi]
                zlo = float(parts[4])
                # If zlo < 100 → bottom region; else → top region
                if zlo < 100.0:
                    n_methanol_bot += count
                else:
                    n_methanol_top += count

        i += 1

    return n_water, n_methanol_bot, n_methanol_top


# ── Read system.xyz ───────────────────────────────────────────────────────────
with open("system.xyz") as f:
    lines = f.readlines()

natoms = int(lines[0].strip())
print(f"Total atoms in system.xyz: {natoms}")

lx, ly, lz = 60.0, 60.0, 200.0

n_water, n_methanol_bot, n_methanol_top = parse_pack_inp("pack.inp")
n_methanol = n_methanol_bot + n_methanol_top

print(f"Parsed from pack.inp:")
print(f"  Water       : {n_water}")
print(f"  Methanol    : {n_methanol}  ({n_methanol_bot} bottom + {n_methanol_top} top)")

expected_atoms = n_water * 3 + n_methanol * 6
if expected_atoms != natoms:
    print(f"WARNING: Expected {expected_atoms} atoms from molecule counts, "
          f"but system.xyz has {natoms} atoms. Check pack.inp.")

# ── Build topology ────────────────────────────────────────────────────────────
atom_id = bond_id = angle_id = 1
mol_id  = 1
atoms   = []
bonds   = []
angles  = []

offset = 2  # skip first 2 header lines of XYZ

# ── Water molecules ──────────────────────────────────────────────────────────
print("Processing water molecules...")
for w in range(n_water):
    idx = offset + w * 3
    o_l  = lines[idx].split()
    h1_l = lines[idx + 1].split()
    h2_l = lines[idx + 2].split()

    o_id = atom_id; h1_id = atom_id + 1; h2_id = atom_id + 2

    atoms.append(f"{o_id}  {mol_id} 1 -0.82  {o_l[1]}  {o_l[2]}  {o_l[3]}")
    atoms.append(f"{h1_id} {mol_id} 2  0.41  {h1_l[1]} {h1_l[2]} {h1_l[3]}")
    atoms.append(f"{h2_id} {mol_id} 2  0.41  {h2_l[1]} {h2_l[2]} {h2_l[3]}")

    bonds.append(f"{bond_id} 1 {o_id} {h1_id}");  bond_id += 1
    bonds.append(f"{bond_id} 1 {o_id} {h2_id}");  bond_id += 1
    angles.append(f"{angle_id} 1 {h1_id} {o_id} {h2_id}"); angle_id += 1

    atom_id += 3
    mol_id  += 1

# ── Methanol molecules ───────────────────────────────────────────────────────
print("Processing methanol molecules...")
offset_methanol = offset + n_water * 3

for m in range(n_methanol):
    idx = offset_methanol + m * 6
    c_l    = lines[idx].split()      # C  (methyl carbon)
    o_l    = lines[idx + 1].split()  # O  (hydroxyl oxygen)
    h_oh_l = lines[idx + 2].split()  # H  (hydroxyl hydrogen)
    h1_l   = lines[idx + 3].split()  # H  (methyl H1)
    h2_l   = lines[idx + 4].split()  # H  (methyl H2)
    h3_l   = lines[idx + 5].split()  # H  (methyl H3)

    c_id   = atom_id;     o_id    = atom_id + 1; h_oh_id = atom_id + 2
    h1_id  = atom_id + 3; h2_id  = atom_id + 4; h3_id   = atom_id + 5

    atoms.append(f"{c_id}    {mol_id} 3 -0.18   {c_l[1]}    {c_l[2]}    {c_l[3]}")
    atoms.append(f"{o_id}    {mol_id} 4 -0.683  {o_l[1]}    {o_l[2]}    {o_l[3]}")
    atoms.append(f"{h_oh_id} {mol_id} 6  0.418  {h_oh_l[1]} {h_oh_l[2]} {h_oh_l[3]}")
    atoms.append(f"{h1_id}   {mol_id} 5  0.145  {h1_l[1]}   {h1_l[2]}   {h1_l[3]}")
    atoms.append(f"{h2_id}   {mol_id} 5  0.145  {h2_l[1]}   {h2_l[2]}   {h2_l[3]}")
    atoms.append(f"{h3_id}   {mol_id} 5  0.145  {h3_l[1]}   {h3_l[2]}   {h3_l[3]}")

    bonds.append(f"{bond_id} 2 {c_id} {o_id}");     bond_id += 1
    bonds.append(f"{bond_id} 3 {o_id} {h_oh_id}");  bond_id += 1
    for hid in [h1_id, h2_id, h3_id]:
        bonds.append(f"{bond_id} 4 {c_id} {hid}");  bond_id += 1

    angles.append(f"{angle_id} 2 {c_id} {o_id} {h_oh_id}"); angle_id += 1
    for hA, hB in [(h1_id, h2_id), (h1_id, h3_id), (h2_id, h3_id)]:
        angles.append(f"{angle_id} 3 {hA} {c_id} {hB}"); angle_id += 1
    for hid in [h1_id, h2_id, h3_id]:
        angles.append(f"{angle_id} 4 {hid} {c_id} {o_id}"); angle_id += 1

    atom_id += 6
    mol_id  += 1

print(f"Total molecules : {mol_id - 1}")
print(f"Total bonds     : {len(bonds)}")
print(f"Total angles    : {len(angles)}")

if atom_id - 1 != natoms:
    print(f"WARNING: created {atom_id - 1} atoms but system.xyz had {natoms}")

# ── Write LAMMPS data file ─────────────────────────────────────────────────
with open("system.data", "w") as f:
    f.write("LAMMPS data file – water/methanol liquid–vapour interface\n\n")
    f.write(f"{natoms} atoms\n")
    f.write(f"{len(bonds)} bonds\n")
    f.write(f"{len(angles)} angles\n")
    f.write("0 dihedrals\n0 impropers\n\n")
    f.write("6 atom types\n4 bond types\n4 angle types\n\n")
    f.write(f"0.0 {lx} xlo xhi\n")
    f.write(f"0.0 {ly} ylo yhi\n")
    f.write(f"0.0 {lz} zlo zhi\n\n")
    f.write("Masses\n\n")
    f.write("1 15.9994  # O  (water,    SPC/Fw)\n")
    f.write("2  1.00794 # H  (water,    SPC/Fw)\n")
    f.write("3 12.0110  # C  (methanol, OPLS-AA)\n")
    f.write("4 15.9994  # O  (methanol, OPLS-AA)\n")
    f.write("5  1.00794 # H  (methyl,   OPLS-AA)\n")
    f.write("6  1.00794 # H  (hydroxyl, OPLS-AA)\n\n")
    f.write("Atoms\n\n")
    for a in atoms:
        f.write(a + "\n")
    f.write("\nBonds\n\n")
    for b in bonds:
        f.write(b + "\n")
    f.write("\nAngles\n\n")
    for a in angles:
        f.write(a + "\n")

print("LAMMPS data file written: system.data")
