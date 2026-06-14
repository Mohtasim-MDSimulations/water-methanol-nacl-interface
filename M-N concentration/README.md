# Water–Methanol–NaCl Liquid–Vapour Interface: Classical MD Simulation

**Thesis:** *Molecular Dynamics Study of Na⁺/Cl⁻ Ions and Methanol at the Water Liquid–Vapour Interface*
**Authors:** Mohtasim · Nishat Siara Ikra
**Supervisor:** Dr. Mohammad Mamun
**Institution:** Bangladesh University of Engineering and Technology (BUET), Dept. of Mechanical Engineering

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scientific Background](#2-scientific-background)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites & Installation](#4-prerequisites--installation)
5. [Quick Start](#5-quick-start)
6. [Step-by-Step Workflow](#6-step-by-step-workflow)
   - 6.1 [Generate Initial Coordinates (PACKMOL)](#61-generate-initial-coordinates-packmol)
   - 6.2 [Convert to LAMMPS Data File](#62-convert-to-lammps-data-file)
   - 6.3 [Run the LAMMPS Simulation](#63-run-the-lammps-simulation)
   - 6.4 [Diagnose the Trajectory](#64-diagnose-the-trajectory)
   - 6.5 [Run Post-Simulation Analysis](#65-run-post-simulation-analysis)
7. [System Definitions](#7-system-definitions)
8. [Force Fields](#8-force-fields)
9. [Simulation Protocol](#9-simulation-protocol)
10. [Analysis Scripts Reference](#10-analysis-scripts-reference)
11. [Output Files Guide](#11-output-files-guide)
12. [Known Issues & Fixes](#12-known-issues--fixes)
13. [Reproducing All Seven Systems (Batch)](#13-reproducing-all-seven-systems-batch)
14. [Citation](#14-citation)

---

## 1. Overview

This repository contains the complete simulation package for a classical molecular dynamics (MD) study of the water liquid–vapour interface in the presence of NaCl ions and methanol co-solvent. The simulations use a 60 × 60 × 200 Å periodic slab geometry at 300 K and produce 5 ns NVE production trajectories from which density profiles, orientational order parameters, surface excess, and hydrogen-bond profiles are extracted.

**Key findings this package reproduces:**
- 40–60 % NaCl-induced enhancement of methanol surface excess (salting-out)
- 10–15 Å ion-depleted interfacial zone
- Setschenow coefficient *k*_s ≈ 0.9 L/mol
- Concentration-dependent water dipole orientation inversion at the interface

---

## 2. Scientific Background

| Quantity | Symbol | Physical meaning |
|---|---|---|
| Gibbs Dividing Surface | GDS | Equimolar surface; located at 50 % bulk water density |
| Surface excess | Γ (mol/Å²) | Excess methanol concentration at the interface vs bulk |
| Setschenow coefficient | *k*_s (L/mol) | Slope of log(Γ/Γ₀) vs [NaCl]; quantifies salting-out |
| Dipole order parameter | ⟨cos θ⟩ | Projection of molecular dipole/bond vector onto z-axis |
| Hydrogen bonds | H-bond | Geometric criterion: r(O–O) < 3.5 Å, ∠(O–H···O) > 150° |

**Relevant theories:** Onsager–Samaras image-charge theory, Born solvation model, Gibbs adsorption isotherm, capillary wave theory.

---

## 3. Repository Structure

```
thesis_simulations/
│
├── README.md                         ← This file
├── .gitignore
│
├── molecule_files/                   ← Input XYZ files for PACKMOL
│   ├── water.xyz
│   ├── methanol.xyz
│   ├── na.xyz
│   └── cl.xyz
│
├── scripts/
│   ├── setup/                        ← Simulation setup scripts
│   │   ├── generate_packmol_input.py ← PACKMOL .inp generator
│   │   ├── xyz2lammps.py             ← XYZ → LAMMPS data file converter
│   │   ├── in.lammps                 ← LAMMPS input script (all systems)
│   │   ├── run_all_simulations.sh    ← Batch runner (all 6 systems)
│   │   └── run_full_analysis.sh      ← Master analysis pipeline runner
│   │
│   ├── analysis/                     ← Post-processing scripts
│   │   ├── analysis_fixed.py         ← Core engine (mol-repair, densities, orientations)
│   │   ├── analyze_all_fixed.py      ← Multi-concentration comparison plots
│   │   ├── surface_excess.py         ← Block-averaged surface excess Γ
│   │   └── hbond_profile.py          ← Hydrogen bond profile (vectorised)
│   │
│   └── utils/
│       └── check_dump.py             ← Trajectory diagnostic tool
│
├── simulations/                      ← Created at runtime; NOT tracked by git
│   ├── sim_N10/
│   ├── sim_M10/
│   ├── sim_M10_N10/
│   ├── sim_M20_N20/
│   ├── sim_M50_N50/
│   └── sim_M100_N100/
│
└── results/                          ← Created by analysis scripts; NOT tracked
    ├── per_system/
    ├── line_graph_fixed/
    └── ...
```

---

## 4. Prerequisites & Installation

### 4.1 Required Software

| Software | Version tested | Purpose | Install notes |
|---|---|---|---|
| **LAMMPS** | 23 Jun 2022 or later | MD engine | See §4.2 |
| **PACKMOL** | 20.x | Initial configuration | See §4.3 |
| **Python** | 3.8 + | Analysis scripts | Usually pre-installed |
| **numpy** | 1.20 + | Array numerics | `pip install numpy` |
| **matplotlib** | 3.4 + | Plotting | `pip install matplotlib` |
| **scipy** | 1.7 + | cKDTree (H-bond script) | `pip install scipy` |
| **MPI** (OpenMPI / MPICH) | any | Parallel LAMMPS | `sudo apt install openmpi-bin` |

> **Windows users:** All steps below assume a Linux/WSL2 environment.
> See `docs/WSL2_LAMMPS_Setup_Guide.md` for a step-by-step WSL2 + LAMMPS
> installation guide tailored to this project.

### 4.2 Installing LAMMPS

```bash
# Install build dependencies
sudo apt update
sudo apt install -y build-essential cmake openmpi-bin libopenmpi-dev \
                    libfftw3-dev python3-dev

# Clone and build LAMMPS with the packages needed for this study
git clone --depth 1 https://github.com/lammps/lammps.git ~/lammps
cd ~/lammps
mkdir build && cd build

cmake ../cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=yes \
  -D PKG_KSPACE=yes \
  -D PKG_MOLECULE=yes \
  -D PKG_RIGID=yes \
  -D PKG_EXTRA-FIX=yes

make -j$(nproc)

# Add to PATH
echo 'export PATH="$HOME/lammps/build:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
lmp_mpi -h | head -5
```

Required LAMMPS packages: `KSPACE` (for PPPM), `MOLECULE` (for atom_style full, bonds, angles), `RIGID` (for SHAKE constraints).

### 4.3 Installing PACKMOL

```bash
# Option A: via apt (Ubuntu 22+)
sudo apt install packmol

# Option B: from source
cd ~
wget https://m3g.github.io/packmol/packmol.tar.gz
tar -xzf packmol.tar.gz
cd packmol
make
sudo cp packmol /usr/local/bin/

# Verify
packmol --version
```

### 4.4 Python dependencies

```bash
pip install numpy matplotlib scipy --break-system-packages
# or inside a venv:
python3 -m venv venv && source venv/bin/activate
pip install numpy matplotlib scipy
```

### 4.5 Clone this repository

```bash
git clone https://github.com/<your-username>/water-methanol-nacl-interface.git
cd water-methanol-nacl-interface
```

---

## 5. Quick Start

To run a **single mixed system** (M10N10: 10 methanol + 10 NaCl pairs) end-to-end:

```bash
# 1. Create simulation directory
mkdir -p simulations/sim_M10_N10
cd simulations/sim_M10_N10

# 2. Copy required files
cp ../../molecule_files/{water.xyz,methanol.xyz,na.xyz,cl.xyz} .
cp ../../scripts/setup/{generate_packmol_input.py,xyz2lammps.py,in.lammps} .

# 3. Generate PACKMOL input (10 methanol, 10 NaCl pairs)
python3 generate_packmol_input.py 10 10

# 4. Build initial structure
packmol < pack.inp > packmol.log 2>&1

# 5. Convert to LAMMPS format
python3 xyz2lammps.py

# 6. Run simulation (6 MPI ranks, ~12–24 h)
mpirun -np 6 lmp_mpi -in in.lammps -log simulation.log

# 7. Check trajectory
python3 ../../scripts/utils/check_dump.py

# 8. Analyse
cd ../..
python3 scripts/analysis/analysis_fixed.py \
    --dump simulations/sim_M10_N10/traj.lammpstrj \
    --nframes 50 --outdir results/per_system
```

To run **all six systems** automatically:

```bash
chmod +x scripts/setup/run_all_simulations.sh
./scripts/setup/run_all_simulations.sh
```

---

## 6. Step-by-Step Workflow

### 6.1 Generate Initial Coordinates (PACKMOL)

`generate_packmol_input.py` writes a `pack.inp` file that places:
- 16,800 water molecules in the slab (z = 30–170 Å)
- Methanol molecules split equally across the two vapor/interface regions
- Na⁺ and Cl⁻ ions inside the water slab only

```bash
python3 scripts/setup/generate_packmol_input.py <n_methanol> <n_nacl>
```

| Argument | Type | Description |
|---|---|---|
| `n_methanol` | int | Total methanol molecules in the system |
| `n_nacl` | int | Number of NaCl ion *pairs* (equal Na⁺ and Cl⁻) |

**Examples:**

```bash
python3 generate_packmol_input.py 0   0     # Pure water
python3 generate_packmol_input.py 0  10     # Pure NaCl baseline (N10)
python3 generate_packmol_input.py 10  0     # Pure methanol baseline (M10)
python3 generate_packmol_input.py 10 10     # Mixed M10N10
python3 generate_packmol_input.py 100 100   # Mixed M100N100
```

Then pack the system:

```bash
packmol < pack.inp > packmol.log 2>&1
# Produces system.xyz (~16,800+ atom XYZ file)
```

> **If PACKMOL fails:** Inspect `packmol.log`. Common causes:
> - Tolerance too tight: edit `tolerance 2.0` in pack.inp → try `2.5`
> - Too many molecules for box size: reduce count or increase box z-dimension in the script

### 6.2 Convert to LAMMPS Data File

```bash
python3 scripts/setup/xyz2lammps.py
# Reads: system.xyz, pack.inp
# Writes: system.data (LAMMPS full-style with bonds, angles, charges)
```

The converter assigns all atom types, partial charges, bonds, and angles automatically. Verify the output with:

```bash
head -30 system.data
```

Expected header:
```
LAMMPS data file – water/methanol/NaCl liquid–vapour interface

50490 atoms        ← depends on system
33600 bonds
16800 angles
...
8 atom types
4 bond types
4 angle types
```

### 6.3 Run the LAMMPS Simulation

```bash
mpirun -np 6 lmp_mpi -in scripts/setup/in.lammps -log simulation.log
```

**Simulation stages inside `in.lammps`:**

| Stage | Fix | Duration | Purpose |
|---|---|---|---|
| 1 | CG minimise | 1000 steps | Remove steric clashes from PACKMOL |
| 2 | NVE | 20 ps | Gentle kinetic energy redistribution |
| 3 | NVT (Nosé-Hoover, τ = 100 fs) | 980 ps | Equilibrate to 300 K |
| 4 | NVE | 5 ns | Production: trajectory + density |

**Output files produced:**

| File | Description |
|---|---|
| `traj.lammpstrj` | Full trajectory (~15–30 GB for 5 ns at 1 ps dump freq.) |
| `traj.xyz` | XYZ trajectory (for visualisation) |
| `density.z.profile` | Running-average 1D density profile from LAMMPS |
| `simulation.log` | Thermo output (temperature, energy, pressure) |
| `final.restart` | LAMMPS restart file |
| `final.data` | Final configuration as LAMMPS data file |

> **Estimated wall-clock time:** 12–24 hours per system on 6 CPU cores.
> Scale `NUM_CORES` in `run_all_simulations.sh` to match your hardware.

### 6.4 Diagnose the Trajectory

Always run this before analysis to catch topology or format issues:

```bash
cd simulations/sim_M10_N10
python3 ../../scripts/utils/check_dump.py --ncheck 2
```

What it checks:

| Check | ✓ Pass | ✗ Fail action |
|---|---|---|
| Required columns (id, type, x, y, z) | Present | Stop — re-run simulation |
| `mol` column | Present | OK — scripts repair from system.data |
| Methanol mol-ID grouping | Non-zero | Silent-zero bug — scripts auto-fix |
| Ions in vapour region | 0 ions | Flag in thesis; check PACKMOL setup |

### 6.5 Run Post-Simulation Analysis

**Option A — Single system:**

```bash
cd simulations/sim_M10_N10
python3 ../../scripts/analysis/analysis_fixed.py \
    --nframes 50 --binwidth 1.0
```

**Option B — All mixed systems comparison:**

```bash
cd <repo_root>
python3 scripts/analysis/analyze_all_fixed.py \
    --nframes 50 --binwidth 1.0 --outdir results/line_graph_fixed
```

**Option C — Full pipeline (all steps):**

```bash
chmod +x scripts/setup/run_full_analysis.sh
./scripts/setup/run_full_analysis.sh
```

---

## 7. System Definitions

Seven atomistic systems are simulated. The naming convention is M*x*N*y* where *x* = methanol count and *y* = NaCl ion pairs.

| Directory | Label | Methanol | NaCl pairs | Total atoms | Purpose |
|---|---|---|---|---|---|
| `sim_N10` | N10 | 0 | 10 | 50,420 | Pure NaCl baseline |
| `sim_M10` | M10 | 10 | 0 | 50,520 | Pure methanol baseline |
| `sim_M10_N10` | M10N10 | 10 | 10 | 50,540 | Low-concentration ternary |
| `sim_M20_N20` | M20N20 | 20 | 20 | 50,600 | — |
| `sim_M50_N50` | M50N50 | 50 | 50 | 50,800 | — |
| `sim_M100_N100` | M100N100 | 100 | 100 | 51,100 | High-concentration ternary |

**Box geometry (all systems):**

```
z = 200 Å  ┌────────────────────┐
            │   Vapour / vacuum  │  z = 170–200 Å
z = 170 Å  ├ ─ ─ GDS (top) ─ ─ ┤
            │                    │
            │    Water slab      │  z = 30–170 Å
            │  (ions confined)   │
z =  30 Å  ├ ─ ─ GDS (bottom) ─ ┤
            │   Vapour / vacuum  │  z = 0–30 Å
z =   0 Å  └────────────────────┘
           x, y: 60 Å (periodic)
```

---

## 8. Force Fields

### Water — SPC/Fw (flexible)

Wu, Y. et al. *J. Chem. Phys.* **124**, 024503 (2006)

| Parameter | Value |
|---|---|
| q(O) | −0.82 e |
| q(H) | +0.41 e |
| σ(O–O) | 3.166 Å |
| ε(O–O) | 0.1554 kcal/mol |
| r₀(O–H) | 1.012 Å |
| k_bond | 4431.5 kcal/mol/Å² |
| θ₀(H–O–H) | 113.24° |
| k_angle | 55.0 kcal/mol/rad² |

O–H bonds and H–O–H angle constrained with SHAKE (tol = 10⁻⁴ Å).

### Methanol — OPLS-AA

Jorgensen, W. L. et al. *J. Am. Chem. Soc.* **118**, 11225 (1996)

| Site | Type | q (e) | σ (Å) | ε (kcal/mol) |
|---|---|---|---|---|
| C (methyl) | CT | −0.18 | 3.50 | 0.0660 |
| O | OH | −0.683 | 3.12 | 0.1700 |
| H (methyl) | HC | +0.145 | 2.50 | 0.0300 |
| H (hydroxyl) | HO | +0.418 | 0.00 | 0.0000 |

### Na⁺ / Cl⁻ — Joung-Cheatham

Joung, I. S. & Cheatham, T. E. *J. Phys. Chem. B* **112**, 9020 (2008)

| Ion | σ (Å) | ε (kcal/mol) | q (e) |
|---|---|---|---|
| Na⁺ | 2.60 | 0.0313098 | +1.0 |
| Cl⁻ | 4.42 | 0.0719407 | −1.0 |

### Combining rules

Geometric mean (Lorentz–Berthelot for LJ, exact charges for Coulombics).
Long-range electrostatics: PPPM (accuracy 10⁻⁴) with Yeh–Berkowitz slab correction (`kspace_modify slab 3.0`).
LJ cutoff: 10 Å.

---

## 9. Simulation Protocol

```
Energy minimisation  →  NVE (20 ps)  →  NVT (980 ps, 300 K)  →  NVE (5 ns, production)
```

| Setting | Value |
|---|---|
| Timestep | 1 fs |
| Thermostat | Nosé-Hoover, τ = 100 fs |
| Integrator | Velocity-Verlet |
| Constraints | SHAKE (O–H bonds, H–O–H angle) |
| Trajectory dump | Every 1 ps (1000 steps) |
| Density profile | Every 1000 steps (LAMMPS ave/chunk) |
| Production frames | 5,000 frames total |
| Analysis frames | Last 50 frames (steady state) |

---

## 10. Analysis Scripts Reference

### `scripts/utils/check_dump.py`

```
python3 check_dump.py [--dump FILE] [--ncheck N]
```

Diagnostic only. Reads N frames and reports column layout, atom counts, mol-ID grouping, and ion z-positions. Run this before any analysis.

---

### `scripts/analysis/analysis_fixed.py`

```
python3 analysis_fixed.py [--dump FILE] [--nframes 50]
                          [--binwidth 1.0] [--outdir fixed_results]
                          [--datafile system.data]
```

**Core engine.** Computes per-system:
- Number density profiles: water (O), methanol (all atoms), Na⁺, Cl⁻
- GDS location (50 % bulk water density criterion)
- Water dipole ⟨cos θ⟩ profile
- Methanol O–H ⟨cos θ⟩ profile
- NaCl nearest-pair ⟨cos θ⟩ profile
- `profiles.csv` (all data in one file)

**Key fix:** if `mol` column is absent in the dump, the script automatically loads molecule IDs from `system.data`, preventing the silent-zero orientation bug.

---

### `scripts/analysis/analyze_all_fixed.py`

```
python3 analyze_all_fixed.py [--nframes 50] [--binwidth 1.0]
                             [--outdir line_graph_fixed]
```

Runs the core engine on all four mixed systems (M10N10–M100N100) and generates:
- Per-system plots in `<outdir>/<sim_name>/`
- Overlay comparison plots in `<outdir>/comparisons/`
- `summary.csv` with all profiles concatenated

---

### `scripts/analysis/surface_excess.py`

```
python3 surface_excess.py [--dump FILE] [--data system.data]
```

Block-averaged surface excess Γ. Splits the 5 ns production run into 5 × 1 ns blocks, computes Γ per block, and reports mean ± std. Outputs:
- `surface_excess_blocks.txt`
- `surface_excess_blocks.pdf/png`

---

### `scripts/analysis/hbond_profile.py`

```
# Single system:
python3 hbond_profile.py [--dump FILE] [--data system.data]
                         [--nframes 50] [--label LABEL]

# All systems from repo root:
python3 hbond_profile.py --run_all [--nframes 50]
```

Vectorised H-bond profile using scipy cKDTree. Geometric criteria: r(O–O) < 3.5 Å, r(H···O) < 3.5 Å, ∠(O–H···O) > 150°. Applies minimum-image convention in x and y.

---

## 11. Output Files Guide

After running the full pipeline, your `results/` directory will contain:

```
results/
├── per_system/
│   ├── sim_M10_N10/
│   │   ├── 01_methanol_density.pdf/.png
│   │   ├── 02_nacl_density.pdf/.png
│   │   ├── 03_na_vs_cl.pdf/.png
│   │   ├── 04_water_orientation.pdf/.png
│   │   ├── 05_methanol_orientation.pdf/.png
│   │   ├── 06_nacl_orientation.pdf/.png
│   │   └── profiles.csv
│   └── sim_M20_N20/ ...  (same structure)
│
└── line_graph_fixed/
    ├── M10_N10/           ← Individual system plots
    ├── M20_N20/
    ├── M50_N50/
    ├── M100_N100/
    ├── comparisons/
    │   ├── comp_methanol_density.pdf/.png
    │   ├── comp_nacl_density.pdf/.png
    │   ├── comp_na_vs_cl.pdf/.png
    │   ├── comp_water_orient.pdf/.png
    │   ├── comp_methanol_orient.pdf/.png
    │   ├── comp_nacl_orient.pdf/.png
    │   └── comp_summary.pdf/.png    ← 2×3 panel figure
    └── summary.csv
```

---

## 12. Known Issues & Fixes

### Issue 1: Missing `mol` column in dump file

**Symptom:** `check_dump.py` reports `✗ 'mol' column: ABSENT`; methanol orientation returns all zeros.

**Cause:** The original `in.lammps` `dump` line did not include `mol` in the column list.

**Fix:** All analysis scripts automatically load molecule IDs from `system.data`. No action needed. To prevent this in future simulations, ensure the dump line reads:
```lammps
dump TRAJ all custom 1000 traj.lammpstrj id mol type x y z vx vy vz
```
This is already correct in the `in.lammps` provided in this repository.

---

### Issue 2: Ions appearing in vapour region

**Symptom:** `check_dump.py` reports `✗ Na⁺ IONS IN VAPOUR`.

**Cause:** Ions placed near the slab boundary may diffuse into the vapour during equilibration if the NVT stage is too short or initial energy is too high.

**Fix:** Run a longer NVE pre-equilibration (increase from 20 ps to 100 ps in `in.lammps`), or reduce the initial velocity magnitude. Ion excursions are also a real physical signal at low concentrations and should be noted rather than suppressed.

---

### Issue 3: PACKMOL fails with "STOP: Packmol exceeded maximum iterations"

**Fix:**
1. Increase tolerance: change `tolerance 2.0` to `tolerance 2.5` in `pack.inp`
2. Reduce molecule count (verify 16,800 water molecules fit the 60×60×140 Å slab)
3. If using Fortran PACKMOL from source, verify it compiled without integer overflow (`make` with gfortran ≥ 8)

---

### Issue 4: LAMMPS "atom lost" / "atoms missing" error

**Symptom:** Simulation crashes with "ERROR: Lost atoms: ..."

**Cause:** Atom coordinates exploded (energy not minimised properly, or bad SHAKE convergence).

**Fix:**
1. Check that minimisation converged: look for `Energy initial, next-to-last, final` in `simulation.log`
2. Reduce timestep to 0.5 fs for the first NVE stage
3. Check that bond coefficients match the actual equilibrium geometry in the XYZ files

---

### Issue 5: matplotlib `Agg` backend error in headless environment

**Symptom:** `UserWarning: Matplotlib is currently using agg, which is a non-GUI backend`

**This is not an error.** All analysis scripts explicitly set `matplotlib.use("Agg")` before any import, which suppresses the interactive display and writes directly to PDF/PNG files. This is the correct behaviour for server/WSL environments.

---

### Issue 6: scipy not found for hbond_profile.py

```bash
pip install scipy --break-system-packages
# or in a venv:
pip install scipy
```

---

## 13. Reproducing All Seven Systems (Batch)

```bash
# From repo root:
chmod +x scripts/setup/run_all_simulations.sh scripts/setup/run_full_analysis.sh

# Step 1: Run all simulations (~3–7 days total on 6 cores)
./scripts/setup/run_all_simulations.sh

# Step 2: Run all analysis
./scripts/setup/run_full_analysis.sh
```

The batch runner creates `simulations/sim_<name>/` directories, copies all required files, runs PACKMOL, converts to LAMMPS format, and launches LAMMPS via MPI. Each system runs sequentially by default. For parallel execution on a cluster with a job scheduler (SLURM/PBS), run each `run_simulation` block as a separate job.

**Disk space estimate:**

| System | traj.lammpstrj | Other outputs | Total |
|---|---|---|---|
| Each system | ~20–30 GB | ~200 MB | ~30 GB |
| All 6 systems | — | — | ~180 GB |

Ensure sufficient disk space before starting the batch run.

---

## 14. Citation

If you use this simulation package or the results derived from it, please cite:

```
Mohtasim, [Nishat Siara Ikra]. "Molecular Dynamics Study of Na⁺/Cl⁻ Ions 
and Methanol at the Water Liquid–Vapour Interface." B.Sc. Thesis, 
Department of Mechanical Engineering, Bangladesh University of Engineering 
and Technology (BUET), 2026. Supervisor: Dr. Mohammad Mamun.
```

**Force-field references:**

- SPC/Fw water: Wu, Y., Tepper, H. L. & Voth, G. A. *J. Chem. Phys.* **124**, 024503 (2006)
- OPLS-AA methanol: Jorgensen, W. L. et al. *J. Am. Chem. Soc.* **118**, 11225 (1996)
- Joung-Cheatham ions: Joung, I. S. & Cheatham, T. E. *J. Phys. Chem. B* **112**, 9020 (2008)
- LAMMPS: Thompson, A. P. et al. *Comput. Phys. Commun.* **271**, 108171 (2022)
- PACKMOL: Martínez, L. et al. *J. Comput. Chem.* **30**, 2157 (2009)

---

*Last updated: June 2026*
