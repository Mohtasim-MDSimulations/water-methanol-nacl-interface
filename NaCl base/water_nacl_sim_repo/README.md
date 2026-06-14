# Water–NaCl Liquid–Vapour Interface: Classical MD Simulation

**System:** Water slab with NaCl ions at the liquid–vapour interface
**Force fields:** SPC/Fw water · Joung-Cheatham ions
**Engine:** LAMMPS + PACKMOL
**Institution:** Bangladesh University of Engineering and Technology (BUET)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Bug Fixes Applied to Original Package](#2-bug-fixes-applied-to-original-package)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites & Installation](#4-prerequisites--installation)
5. [Quick Start](#5-quick-start)
6. [Step-by-Step Workflow](#6-step-by-step-workflow)
7. [System Definition](#7-system-definition)
8. [Force Fields](#8-force-fields)
9. [Simulation Protocol](#9-simulation-protocol)
10. [Analysis Scripts Reference](#10-analysis-scripts-reference)
11. [Output Files Guide](#11-output-files-guide)
12. [Known Issues & Fixes](#12-known-issues--fixes)
13. [Running Different Ion Concentrations](#13-running-different-ion-concentrations)
14. [Citation](#14-citation)

---

## 1. Overview

This package simulates a 60 × 60 × 200 Å slab of liquid water containing NaCl
ions at 300 K. The two liquid–vapour interfaces develop naturally from the slab
geometry. A 5 ns NVT production run yields ion density profiles, water dipole
orientation, surface tension, and radial distribution functions.

**Key observables:**
- Na⁺ and Cl⁻ number density profiles (depletion from the interface)
- Water dipole ⟨cos θ⟩ orientation across the interface
- Surface tension γ from pressure-tensor anisotropy
- O–O and Na–Cl radial distribution functions

---

## 2. Bug Fixes Applied to Original Package

The original package contained **nine bugs**, including a physically critical
double-correction of electrostatics and an invalid surface-tension "scaling"
that hid real simulation errors. All are corrected here.

| # | File | Location | Bug | Fix |
|---|------|----------|-----|-----|
| 1 | `equilibration.in` `production.in` | `boundary` + `kspace` | **Critical: double electrostatic correction.** `boundary p p f` (non-periodic z) was combined with `kspace_modify slab 3.0`. The slab correction is designed exclusively for `p p p` geometry. With `p p f`, no periodic z-image exists, so the correction is unnecessary and produces wrong Coulomb forces — interfacial structure, ion positions, and surface tension are all affected. | Changed to `boundary p p p` + retain `kspace_modify slab 3.0` |
| 2 | `production.in` | `fix walls` | **Wall potentials present during production.** `wall/lj126` forces at z-boundaries suppress capillary-wave fluctuations, bias the interfacial density profile, and contribute a spurious energy term to the pressure tensor, corrupting the surface tension. | Walls removed from production; used only (optionally) during equilibration if needed |
| 3 | `equilibration.in` `production.in` | `special_bonds` | **Wrong special_bonds setting.** `special_bonds lj/coul 0.0 0.0 1.0` sets the 1-4 non-bonded scaling factor to 1.0. SPC/Fw is a 3-atom molecule with no 1-4 pairs, so this has no numerical effect — but the correct and unambiguous setting for SPC water is `0 0 0` (all intramolecular non-bonded excluded). | Changed to `special_bonds lj/coul 0 0 0` |
| 4 | `pdb_to_lammps.py` | mol-ID assignment | **Fragile positional mol-ID logic.** Original code inferred molecule boundaries from atom ordering in the PDB (assumed O always before H). PACKMOL typically writes them in order, but this assumption is undocumented and breaks if the PDB is reordered. | Replaced with explicit (residue_name, residue_serial) key-based grouping — order-independent |
| 5 | `final_analysis_complete.py` | `read_surface_tension_and_correct()` | **Critical: invalid surface tension "correction".** Script computed `scaling = 72.0 / mean_raw` then multiplied all γ values by it. This forced the last-ns mean to equal the textbook bulk water value (72 mN/m) regardless of what the simulation produced, hiding any setup error. SPC/Fw water actually gives ~63 mN/m at 300 K. | Removed scaling entirely; raw γ reported with reference line at 63 mN/m |
| 6 | `final_analysis_complete.py` | `time_ns` calculation | **Timestep offset bug.** Production run inherited the equilibration timestep counter (~500,000). Analysis computed `time_ns = timestep × 1e-6` with no offset, placing 5 ns of production at 0.5–5.5 ns. The 4–5 ns analysis window then picked up frames from production ns 3.5–4.5. | Added `reset_timestep 0` in `production.in`; analysis uses `--prod_start 0` |
| 7 | `final_analysis_complete.py` | `process_frame_*` | **Positional atom indexing.** Assumed LAMMPS writes atoms in sorted mol-ID / creation order. MPI decomposition does not guarantee this. | Replaced with mol-ID dictionary grouping (same fix as in the water-methanol package) |
| 8 | `production.in` | `compute rdf_O_O` | **Wrong RDF group.** `compute rdf_O_O oxygen rdf 1000 1 1` uses the `oxygen` group but type args `1 1`. LAMMPS RDF pair counting uses the whole system for normalization unless the group is `all`; using a restricted group gives incorrect g(r) normalization. | Changed to `compute rdf_O_O all rdf 1000 1 1` |
| 9 | `equilibration.in` | `bond_coeff` r₀ | **Wrong O–H equilibrium bond length.** `bond_coeff 1 1059.162 1.0` uses r₀ = 1.0 Å. SPC/Fw (Wu et al., 2006) specifies r₀ = 1.012 Å. The 1.2% discrepancy shifts the O–H stretch frequency and introduces a systematic error in water density. | Changed to `bond_coeff 1 1059.162 1.012` |

---

## 3. Repository Structure

```
water_nacl_sim_repo/
│
├── README.md
├── .gitignore
│
├── molecule_files/
│   ├── water.pdb             ← SPC/Fw geometry (O–H bond: 1.012 Å corrected)
│   ├── na.pdb                ← Na⁺ ion
│   └── cl.pdb                ← Cl⁻ ion
│
├── scripts/
│   ├── setup/
│   │   ├── generate_packmol_input.py  ← Parametric PACKMOL input generator
│   │   ├── pdb_to_lammps.py           ← PDB → LAMMPS data converter (mol-ID fixed)
│   │   ├── equilibration.in           ← LAMMPS equilibration script (corrected)
│   │   ├── production.in              ← LAMMPS production script (corrected)
│   │   └── run_simulation.sh          ← End-to-end runner
│   │
│   ├── analysis/
│   │   └── final_analysis_complete.py ← Complete analysis (all plots)
│   │
│   └── utils/
│       └── check_dump.py              ← Trajectory diagnostic tool
│
└── simulations/                       ← Created at runtime; NOT in git
    └── sim_N10/
        ├── water_slab.pdb
        ├── water_interface.data
        ├── production.lammpstrj
        └── ...
```

---

## 4. Prerequisites & Installation

### Required software

| Software | Version tested | Purpose |
|----------|---------------|---------|
| LAMMPS | 23 Jun 2022+ | MD engine |
| PACKMOL | 20.x | Initial structure |
| Python | 3.8+ | Analysis |
| numpy | 1.20+ | Numerics |
| matplotlib | 3.4+ | Plotting |
| OpenMPI / MPICH | any | Parallel LAMMPS |

### Quick install (Ubuntu/WSL2)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git cmake python3 python3-pip \
                    mpich libfftw3-dev

# LAMMPS
cd $HOME
git clone --branch stable https://github.com/lammps/lammps.git
cd lammps && mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_MPI=yes \
      -D PKG_MOLECULE=yes -D PKG_KSPACE=yes -D PKG_RIGID=yes \
      ../cmake
make -j$(nproc) && sudo make install

# PACKMOL
cd $HOME
wget https://github.com/m3g/packmol/archive/refs/tags/v20.0.0.tar.gz
tar -xzf v20.0.0.tar.gz && cd packmol-20.0.0
./configure && make && sudo make install

# Python
pip3 install numpy matplotlib

# Verify
lmp -version && packmol < /dev/null; python3 -c "import numpy,matplotlib; print('OK')"
```

### Clone this repository

```bash
git clone https://github.com/<your-username>/water-nacl-interface.git
cd water-nacl-interface
```

---

## 5. Quick Start

Full simulation with 10 NaCl ion pairs (~30 h on 4 cores):

```bash
chmod +x scripts/setup/run_simulation.sh
./scripts/setup/run_simulation.sh 10 4
# Args: n_nacl_pairs (default 10), n_cores (default 4)
```

Results appear in `results/sim_N10/`.

---

## 6. Step-by-Step Workflow

### Step 1 — Generate PACKMOL input

```bash
python3 scripts/setup/generate_packmol_input.py <n_nacl_pairs>
```

Places 16,632 water molecules and n_nacl Na⁺ + n_nacl Cl⁻ ions inside the
slab (z = 30–170 Å). Both ion types go into the slab — they are strongly
solvated and have no physical presence in the vapour phase.

### Step 2 — Build initial structure

```bash
packmol < water_slab.inp > packmol.log 2>&1
```

Produces `water_slab.pdb`. Check `packmol.log` for `Packing solved`.

### Step 3 — Convert to LAMMPS format

```bash
python3 scripts/setup/pdb_to_lammps.py water_slab.pdb water_interface.data
```

Produces `water_interface.data` with full topology. Mol-IDs are assigned from
PDB residue serial numbers — independent of atom ordering within each residue.

Verify:

```bash
head -15 water_interface.data
```

Expected:

```
# LAMMPS data file – water/NaCl liquid-vapour interface

50016 atoms      ← 16632*3 + 10 + 10 for N10
33264 bonds
16632 angles
...
4 atom types
```

### Step 4 — Run equilibration (~4 h on 4 cores)

```bash
mpirun -np 4 lmp -in scripts/setup/equilibration.in -log equilibration.log
```

Stages: CG minimisation → NVE+Langevin (100 ps) → NVT Nosé-Hoover (400 ps).

### Step 5 — Run production (~30 h on 4 cores)

```bash
mpirun -np 4 lmp -in scripts/setup/production.in -log production.log
```

Writes trajectory every 10 ps, density profiles every 1 ps, surface tension
every 1 ps (averaged over 10 samples per output).

### Step 6 — Check trajectory

```bash
python3 scripts/utils/check_dump.py --dump production.lammpstrj
```

Confirms mol column present, ion z-ranges within slab, timestep starts at 0.

### Step 7 — Run analysis

```bash
python3 scripts/analysis/final_analysis_complete.py \
    --traj  production.lammpstrj \
    --st    surface_tension.dat \
    --outdir results/
```

---

## 7. System Definition

```
z = 200 Å  ┌────────────────────────────────┐
            │        Vapour / vacuum          │  z = 170–200 Å
z = 170 Å  ├ ─ ─ ─ ─ GDS (top) ─ ─ ─ ─ ─ ─ ┤
            │                                 │
            │     Water slab + NaCl ions      │  z = 30–170 Å
            │       (16,632 H₂O)              │
            │                                 │
z =  30 Å  ├ ─ ─ ─ GDS (bottom) ─ ─ ─ ─ ─ ─ ┤
            │        Vapour / vacuum          │  z = 0–30 Å
z =   0 Å  └────────────────────────────────┘
                  x, y: 60 Å (periodic)
```

| Property | Value |
|----------|-------|
| Box dimensions | 60 × 60 × 200 Å |
| Water molecules | 16,632 |
| NaCl ion pairs (default) | 10 |
| Total atoms (N10) | 50,016 |
| Temperature | 300 K |
| Boundary | p p p (fully periodic + slab correction) |

---

## 8. Force Fields

### Water — SPC/Fw (flexible, no SHAKE required)

Wu, Y. et al. *J. Chem. Phys.* **124**, 024503 (2006)

| Parameter | Value |
|-----------|-------|
| q(O) | −0.82 e |
| q(H) | +0.41 e |
| ε(O–O) | 0.1554 kcal/mol |
| σ(O–O) | 3.165 Å |
| r₀(O–H) | **1.012 Å** ← corrected from 1.0 |
| k_bond | 1059.162 kcal/mol/Å² |
| θ₀(H–O–H) | 113.24° |
| k_angle | 75.90 kcal/mol/rad² |

SPC/Fw is a *flexible* model — no SHAKE constraints are used. Bonds and angles
are treated as harmonic potentials.

### Na⁺ / Cl⁻ — Joung-Cheatham

Joung, I. S. & Cheatham, T. E. *J. Phys. Chem. B* **112**, 9020 (2008)

| Ion | σ (Å) | ε (kcal/mol) | q (e) |
|-----|--------|--------------|-------|
| Na⁺ | 2.159 | 0.1684 | +1.0 |
| Cl⁻ | 4.830 | 0.0127 | −1.0 |

### Combining rules

All Lennard-Jones cross terms (O–H, O–Na, O–Cl, H–Na, H–Cl, Na–Cl) are given
explicitly via `pair_coeff` and are **not** derived from a mixing rule —
H cross-terms are zero (H carries no LJ site in SPC/Fw), and the ion–water
cross terms reflect the original Joung-Cheatham parameterisation for SPC-family
water rather than a simple arithmetic or geometric mean. `pair_modify mix
arithmetic` is set as a harmless default but is never actually invoked since
every pair has an explicit coefficient.

Long-range electrostatics: PPPM (accuracy 10⁻⁴) with Yeh–Berkowitz slab
correction (`kspace_modify slab 3.0`), applied under `boundary p p p`.

---

## 9. Simulation Protocol

```
CG minimise  →  NVE+Langevin 100 ps  →  NVT 400 ps  →  NVT production 5 ns
```

| Setting | Value |
|---------|-------|
| Timestep | 1 fs |
| Thermostat | Nosé-Hoover, τ = 100 fs |
| SHAKE | None (SPC/Fw is flexible) |
| Production trajectory dump | Every 10 ps (10,000 steps) |
| Density profiles | LAMMPS ave/chunk every 1 ps |
| Surface tension output | Running average every 1 ps |
| Production length | 5 ns (5,000,000 steps) |
| Analysis windows | 3–4 ns (4th ns) and 4–5 ns (5th ns) |

**Note on surface tension:** The raw γ from SPC/Fw at 300 K is ~63–65 mN/m
(lower than the experimental 72 mN/m). This is an expected force-field
limitation. The analysis script reports raw γ and draws a reference line at
63 mN/m. Do NOT apply a post-hoc scaling factor.

---

## 10. Analysis Scripts Reference

### `scripts/utils/check_dump.py`

```
python3 check_dump.py [--dump FILE] [--ncheck N]
```

Reports column layout, atom-type counts, mol grouping, and ion z-ranges.
Run before analysis. Confirms timestep starts at 0 (from `reset_timestep 0`).

---

### `scripts/analysis/final_analysis_complete.py`

```
python3 final_analysis_complete.py [--traj FILE] [--st FILE]
                                   [--outdir DIR] [--prod_start 0]
                                   [--nframes_max 0]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--traj` | `production.lammpstrj` | Trajectory file |
| `--st` | `surface_tension.dat` | LAMMPS surface tension output |
| `--outdir` | `results` | Output directory |
| `--prod_start` | `0` | First production timestep (0 with `reset_timestep 0`) |
| `--nframes_max` | `0` | Max frames to read (0 = all) |

**Outputs:**

| File | Contents |
|------|----------|
| `surface_tension_bars_raw.pdf/png` | Block-averaged γ (1 ns windows, raw) |
| `surface_tension_blocks.txt` | Tabulated γ values |
| `density_na_5ns_line.pdf/png` | Na⁺ density – 4–5 ns |
| `density_cl_5ns_line.pdf/png` | Cl⁻ density – 4–5 ns |
| `density_na_cl_5ns_line.pdf/png` | Na⁺ + Cl⁻ overlaid – 4–5 ns |
| `density_na_cl_5ns_bars.pdf/png` | Ion density bar chart |
| `orientation_4ns_line.pdf/png` | Water ⟨cos θ⟩ – 3–4 ns |
| `orientation_4ns_bars.pdf/png` | Same, bar format |
| `orientation_5ns_line.pdf/png` | Water ⟨cos θ⟩ – 4–5 ns |
| `orientation_5ns_bars.pdf/png` | Same, bar format |
| `orientation_comparison_line.pdf/png` | Both windows overlaid |
| `orientation_comparison_bars.pdf/png` | Same, bar format |
| `ion_density_profiles_5ns.csv` | Na⁺ and Cl⁻ profiles tabulated |
| `water_orientation_4ns.txt` | ⟨cos θ⟩ profile at 3–4 ns |
| `water_orientation_5ns.txt` | ⟨cos θ⟩ profile at 4–5 ns |

---

## 11. Output Files Guide

After a full run, `simulations/sim_N10/` contains:

```
sim_N10/
├── water_slab.pdb
├── water_interface.data
├── equilibration.lammpstrj     ← Equilibration trajectory
├── equilibration_final.restart ← Used to start production
├── production.lammpstrj        ← Production trajectory (~5–10 GB)
├── production.log
├── density_O.profile           ← LAMMPS running-average O density
├── density_Na.profile
├── density_Cl.profile
├── surface_tension.dat         ← γ time series
├── rdf_O_O.dat                 ← O–O RDF
├── rdf_Na_Cl.dat               ← Na–Cl RDF
└── results/
    ├── surface_tension_bars_raw.pdf/.png
    ├── density_na_5ns_line.pdf/.png
    ├── density_cl_5ns_line.pdf/.png
    ├── density_na_cl_5ns_line.pdf/.png
    ├── density_na_cl_5ns_bars.pdf/.png
    ├── orientation_4ns_line.pdf/.png
    ├── orientation_4ns_bars.pdf/.png
    ├── orientation_5ns_line.pdf/.png
    ├── orientation_5ns_bars.pdf/.png
    ├── orientation_comparison_line.pdf/.png
    ├── orientation_comparison_bars.pdf/.png
    ├── ion_density_profiles_5ns.csv
    ├── water_orientation_4ns.txt
    └── water_orientation_5ns.txt
```

---

## 12. Known Issues & Fixes

### LAMMPS reports "lost atoms" during equilibration

**Cause:** Close contacts in PACKMOL output not resolved by minimisation.

**Fix:**
1. Try `tolerance 2.5` in place of `tolerance 2.0` in `water_slab.inp`
2. Increase minimisation iterations: `minimize 1.0e-4 1.0e-6 5000 50000`
3. Run a longer NVE+Langevin stage (increase from 100,000 to 200,000 steps)

### Surface tension is unexpectedly negative

**Cause:** Slab not in equilibrium; pzz fluctuations dominate early frames.
The corrected `production.in` uses `reset_timestep 0`, so the first 3 ns
(0–3 ns) is treated as additional equilibration in the analysis.

**Expected value:** SPC/Fw gives γ ≈ 63–65 mN/m at 300 K (not 72 mN/m).

### Analysis finds 0 ion frames

**Cause:** `--prod_start` mismatch. If you used an old trajectory that did
NOT have `reset_timestep 0`, the production timesteps start at ~500,000.
Pass `--prod_start 500000`.

### PPPM accuracy warning

```
PPPM: accuracy is too small ...
```

Reduce accuracy to `1.0e-3` in `kspace_style pppm` if memory is limited,
or increase the PPPM order with `kspace_modify order 6`.

### matplotlib Agg backend warning

Not an error. All scripts use `matplotlib.use("Agg")` explicitly for
headless (no-display) environments. PDF and PNG files are written correctly.

---

## 13. Running Different Ion Concentrations

The workflow is fully parametric:

```bash
# 10 ion pairs (default, ~0.1 mol/kg)
./scripts/setup/run_simulation.sh 10 4

# 50 ion pairs (~0.5 mol/kg)
./scripts/setup/run_simulation.sh 50 4

# 100 ion pairs (~1.0 mol/kg)
./scripts/setup/run_simulation.sh 100 4
```

Each run creates `simulations/sim_N<n>/` with all files isolated.
Results go into `results/sim_N<n>/`.

---

## 14. Citation

If you use this package, please cite:

```
[Author names]. "Water-NaCl Liquid-Vapour Interface MD Simulation Package."
Bangladesh University of Engineering and Technology (BUET), 2026.
```

**Force-field references:**

- SPC/Fw water: Wu, Y., Tepper, H. L. & Voth, G. A. *J. Chem. Phys.* **124**, 024503 (2006)
- Joung-Cheatham ions: Joung, I. S. & Cheatham, T. E. *J. Phys. Chem. B* **112**, 9020 (2008)
- LAMMPS: Thompson, A. P. et al. *Comput. Phys. Commun.* **271**, 108171 (2022)
- PACKMOL: Martínez, L. et al. *J. Comput. Chem.* **30**, 2157 (2009)
- Yeh–Berkowitz slab correction: Yeh, I.-C. & Berkowitz, M. L. *J. Chem. Phys.* **111**, 3155 (1999)

---

*Last updated: June 2026*
