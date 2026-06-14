#!/usr/bin/env python3
"""
pdb_to_lammps.py
================
Convert a PACKMOL-generated water_slab.pdb into a LAMMPS full-style data
file (water_interface.data) with bonds, angles, charges, and masses.

Atom types:
  1 – O  (water,    SPC/Fw)   charge: −0.82 e,  mass: 15.9994
  2 – H  (water,    SPC/Fw)   charge: +0.41 e,  mass:  1.008
  3 – Na⁺ (Joung-Cheatham)    charge: +1.00 e,  mass: 22.98977
  4 – Cl⁻ (Joung-Cheatham)    charge: −1.00 e,  mass: 35.453

Bond types:
  1 – O–H  (SPC/Fw,  k = 1059.162 kcal/mol/Å², r₀ = 1.012 Å)

Angle types:
  1 – H–O–H (SPC/Fw,  k = 75.90 kcal/mol/rad², θ₀ = 113.24°)

BUG FIX vs original:
  Original mol-ID assignment read atoms positionally and assumed O always
  appears before H in the PDB. Replaced with explicit residue-serial–based
  grouping: each PDB residue (WAT, NA, CL) gets its own mol-ID, independent
  of atom ordering within the file.

Usage (run inside the simulation work directory, alongside water_slab.pdb):
  python3 pdb_to_lammps.py [water_slab.pdb] [water_interface.data]
"""
import sys
import os

CHARGE_O  = -0.82
CHARGE_H  =  0.41
CHARGE_NA =  1.00
CHARGE_CL = -1.00

# Bond: SPC/Fw  (BUG FIX: r0 corrected from 1.0 to 1.012 Å)
BOND_K  = 1059.162
BOND_R0 = 1.012      # BUG FIX: original used 1.0 — correct SPC/Fw value is 1.012

ANGLE_K     = 75.90
ANGLE_THETA = 113.24

MASSES = {1: 15.9994, 2: 1.008, 3: 22.98977, 4: 35.453}

BOX = (60.0, 60.0, 200.0)


def read_pdb(filename: str):
    """
    Parse a PACKMOL PDB file and return a list of atom dicts.

    BUG FIX: uses (residue_name, residue_serial) as the grouping key for
    mol-IDs instead of positional inference, so order-of-appearance within
    each residue does not matter.
    """
    raw_atoms = []
    with open(filename) as f:
        for line in f:
            if not line.startswith(("HETATM", "ATOM")):
                continue
            name      = line[12:16].strip()
            res_name  = line[17:20].strip()
            # PDB residue serial (columns 23-26, 1-indexed)
            try:
                res_serial = int(line[22:26].strip())
            except ValueError:
                res_serial = 0
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            raw_atoms.append({
                "name":       name,
                "res_name":   res_name,
                "res_serial": res_serial,
                "x": x, "y": y, "z": z,
            })
    return raw_atoms


def assign_mol_ids(raw_atoms):
    """
    Assign unique mol-IDs using (res_name, res_serial) as the key.
    Returns list of atoms with 'mol', 'lammps_type' fields added.
    """
    # Map (res_name, res_serial) → mol_id
    seen  = {}
    next_mol = 1
    atoms = []

    for a in raw_atoms:
        key = (a["res_name"], a["res_serial"])
        if key not in seen:
            seen[key] = next_mol
            next_mol += 1
        mol_id = seen[key]

        res = a["res_name"]
        name = a["name"]

        if res == "WAT":
            ltype = 1 if name == "O" else 2
        elif res == "NA":
            ltype = 3
        elif res == "CL":
            ltype = 4
        else:
            continue   # skip unknown residues

        atoms.append({**a, "mol": mol_id, "ltype": ltype})

    return atoms


def write_lammps_data(atoms, output_file: str) -> None:
    """Write LAMMPS full-style data file."""
    lx, ly, lz = BOX

    # Separate by type
    water_mols = {}   # mol_id → {"O": atom, "H": [atom, atom]}
    na_atoms   = []
    cl_atoms   = []

    for a in atoms:
        if a["res_name"] == "WAT":
            water_mols.setdefault(a["mol"], {"O": None, "H": []})
            if a["ltype"] == 1:
                water_mols[a["mol"]]["O"] = a
            else:
                water_mols[a["mol"]]["H"].append(a)
        elif a["res_name"] == "NA":
            na_atoms.append(a)
        elif a["res_name"] == "CL":
            cl_atoms.append(a)

    n_water = len(water_mols)
    n_na    = len(na_atoms)
    n_cl    = len(cl_atoms)
    n_atoms = n_water * 3 + n_na + n_cl
    n_bonds  = n_water * 2
    n_angles = n_water

    print(f"Writing LAMMPS data file:")
    print(f"  Water molecules : {n_water}")
    print(f"  Na+ ions        : {n_na}")
    print(f"  Cl- ions        : {n_cl}")
    print(f"  Total atoms     : {n_atoms}")

    with open(output_file, "w") as f:
        f.write("# LAMMPS data file – water/NaCl liquid-vapour interface\n\n")
        f.write(f"{n_atoms} atoms\n{n_bonds} bonds\n{n_angles} angles\n")
        f.write("0 dihedrals\n0 impropers\n\n")
        f.write("4 atom types\n1 bond types\n1 angle types\n\n")
        f.write(f"0.0 {lx} xlo xhi\n0.0 {ly} ylo yhi\n0.0 {lz} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 15.9994   # O  (water)\n")
        f.write("2  1.008    # H  (water)\n")
        f.write("3 22.98977  # Na+\n")
        f.write("4 35.453    # Cl-\n\n")
        f.write("Atoms # full\n\n")

        aid = 1
        bonds_list   = []
        angles_list  = []
        bond_id      = 1
        angle_id     = 1

        # ── Water molecules ───────────────────────────────────────────────────
        for mol_id in sorted(water_mols.keys()):
            wm = water_mols[mol_id]
            if wm["O"] is None or len(wm["H"]) < 2:
                print(f"  WARNING: incomplete water molecule mol_id={mol_id}, skipping")
                continue
            o  = wm["O"]
            h1 = wm["H"][0]
            h2 = wm["H"][1]

            o_aid  = aid
            h1_aid = aid + 1
            h2_aid = aid + 2

            f.write(f"{o_aid}  {mol_id} 1 {CHARGE_O:.4f}  "
                    f"{o['x']:.6f} {o['y']:.6f} {o['z']:.6f}\n")
            f.write(f"{h1_aid} {mol_id} 2 {CHARGE_H:.4f}  "
                    f"{h1['x']:.6f} {h1['y']:.6f} {h1['z']:.6f}\n")
            f.write(f"{h2_aid} {mol_id} 2 {CHARGE_H:.4f}  "
                    f"{h2['x']:.6f} {h2['y']:.6f} {h2['z']:.6f}\n")

            bonds_list.append(f"{bond_id}   1  {o_aid} {h1_aid}")
            bond_id += 1
            bonds_list.append(f"{bond_id}   1  {o_aid} {h2_aid}")
            bond_id += 1
            angles_list.append(f"{angle_id}  1  {h1_aid} {o_aid} {h2_aid}")
            angle_id += 1

            aid += 3

        # ── Na⁺ ions ──────────────────────────────────────────────────────────
        max_water_mol = max(water_mols.keys()) if water_mols else 0
        ion_mol = max_water_mol + 1

        for na in na_atoms:
            f.write(f"{aid} {ion_mol} 3 {CHARGE_NA:.4f}  "
                    f"{na['x']:.6f} {na['y']:.6f} {na['z']:.6f}\n")
            aid      += 1
            ion_mol  += 1

        # ── Cl⁻ ions ──────────────────────────────────────────────────────────
        for cl in cl_atoms:
            f.write(f"{aid} {ion_mol} 4 {CHARGE_CL:.4f}  "
                    f"{cl['x']:.6f} {cl['y']:.6f} {cl['z']:.6f}\n")
            aid      += 1
            ion_mol  += 1

        # ── Bonds ─────────────────────────────────────────────────────────────
        f.write("\nBonds\n\n")
        for b in bonds_list:
            f.write(b + "\n")

        # ── Angles ────────────────────────────────────────────────────────────
        f.write("\nAngles\n\n")
        for a_line in angles_list:
            f.write(a_line + "\n")

    print(f"  Written: {output_file}")


def main():
    in_file  = sys.argv[1] if len(sys.argv) > 1 else "water_slab.pdb"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "water_interface.data"

    if not os.path.exists(in_file):
        sys.exit(f"ERROR: {in_file} not found. Run PACKMOL first.")

    print(f"Reading: {in_file}")
    raw_atoms = read_pdb(in_file)
    print(f"  Raw atoms parsed: {len(raw_atoms)}")

    atoms = assign_mol_ids(raw_atoms)
    print(f"  Atoms with mol-IDs: {len(atoms)}")

    write_lammps_data(atoms, out_file)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
