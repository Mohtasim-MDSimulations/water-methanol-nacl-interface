# Water–Methanol Liquid–Vapour Interface: Classical MD Simulation

**System:** Water slab with methanol at the liquid–vapour interface
**Force fields:** SPC/Fw water · OPLS-AA methanol
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
13. [Running Different Methanol Counts](#13-running-different-methanol-counts)
14. [Citation](#14-citation)

---

## 1. Overview

This package simulates a 60 × 60 × 200 Å slab of liquid water at 300 K with
methanol molecules adsorbing at the two liquid–vapour interfaces. It produces
5 ns NVE trajectories from which density profiles, orientational order parameters,
and surface tension are extracted.

**Key observables:**
- Water dipole ⟨cos θ⟩ profile across the interface
- Methanol O–H ⟨cos θ⟩ orientation at both interfaces
- Methanol number and mass density profiles
- Surface tension γ from pressure-tensor anisotropy

---

## 2. Bug Fixes Applied to Original Package

The original simulation package contained **ten bugs** ranging from silent numerical
errors to physically incorrect electrostatics. All are corrected in this repository.

| # | File | Location | Bug | Fix |
|---|------|----------|-----|-----|
| 1 | `in.lammps` | `kspace_style` | **Yeh–Berkowitz slab correction missing.** `kspace_modify slab 3.0` was absent. PPPM in fully periodic 3D creates an artefact electric field across the vacuum gap, corrupting interfacial structure and surface tension. | Added `kspace_modify slab 3.0` |
| 2 | `in.lammps` | `fix SHAKE` | **Methanol O–H bond not constrained.** Original: `b 1 a 1` (water O–H and angle only). Methanol O–H bond (bond type 3) was left flexible, causing slow energy drift over 5 ns. | Changed to `b 1 3 a 1` |
| 3 | `in.lammps` | `pair_coeff` | **Inconsistent mixing rule.** Original manually listed all cross pair_coeff using arithmetic-mean σ (Lorentz–Berthelot) but geometric-mean ε. This hybrid is not a valid combining rule and differs from both OPLS-AA and SPC/Fw conventions. | Removed all manual cross terms; added `pair_modify mix geometric` |
| 4 | `in.lammps` | `dump` | **No mol column in trajectory.** Original dump: `id type x y z vx vy vz q`. Without mol, analysis scripts must use positional indexing, which is wrong under MPI. | Added `mol` to dump columns |
| 5 | `analysis.py` | `process_frame()` | **cos θ calculated incorrectly.** The walrus operators `h_mid_x := (h1_x+h2_x)/2.0 - o_x` and `h_mid_y := ...` were embedded inside `np.sqrt()` but the subtraction of `o_x`/`o_y` was applied *after* squaring, not before. The x and y contributions to the dipole norm were systematically wrong, giving an incorrect ⟨cos θ⟩. | Rewrote using explicit 3D numpy vector: `dipole = h_mid - o_pos; norm = np.linalg.norm(dipole)` |
| 6 | `analysis.py` | `process_frame()` | **Positional atom indexing.** Used `atoms_arr[mol*3]` to locate water molecules, assuming LAMMPS writes atoms in creation order. MPI decomposition breaks this. | Replaced with mol-ID grouping from dump |
| 7 | `final_analysis_methanol.py` | `process_frame_methanol()` | **Same positional indexing bug** with fixed `TOTAL_WATER_ATOMS` offset for methanol. | Replaced with mol-ID grouping |
| 8 | `pack.inp` | methanol placement | **All methanol placed in top vapour region only** (z = 170–200 Å). Bottom interface started with zero methanol, biasing adsorption asymmetrically from t = 0. | `generate_packmol_input.py` splits molecules equally between top and bottom |
| 9 | `xyz2lammps.py` | constants | **Hardcoded `nwater=16800` and `nmethanol=10`.** Changing pack.inp silently produced wrong topology. | Script now parses pack.inp dynamically |
| 10 | `analysis.py` | `read_lammps_dump()` | **Charge column accessed by default index 8** via `col.get('q', 8)`. Charges are static; reading them from the dump is unnecessary, and a wrong default index causes silent misreads if column order differs. | Removed charge column entirely from dump reader |

---

## 3. Repository Structure

```
water_methanol_sim_repo/
│
├── README.md
├── .gitignore
│
├── molecule_files/
│   ├── water.xyz             ← SPC/Fw O-H-H geometry
│   └── methanol.xyz          ← OPLS-AA C-O-H-H-H-H geometry
│
├── scripts/
│   ├── setup/
│   │   ├── generate_packmol_input.py  ← Parametric PACKMOL input generator
│   │   ├── xyz2lammps.py              ← XYZ → LAMMPS data file converter
│   │   ├── in.lammps                  ← LAMMPS input script (corrected)
│   │   └── run_simulation.sh          ← End-to-end batch runner
│   │
│   ├── analysis/
│   │   ├── analysis.py                ← Density + orientation + surface tension
│   │   └── final_analysis_methanol.py ← Methanol-focused analysis
│   │
│   └── utils/
│       └── check_dump.py              ← Trajectory diagnostic tool
│
└── simulations/                       ← Created at runtime; NOT in git
    └── sim_M10/
        ├── traj.lammpstrj
        ├── simulation.log
        └── ...
```

---

## 4. Prerequisites & Installation

### Required software

| Software | Version tested | Purpose |
|----------|---------------|---------|
| LAMMPS (MPI) | 23 Jun 2022+ | MD engine |
| PACKMOL | 20.x | Initial configuration |
| Python | 3.8+ | Analysis |
| numpy | 1.20+ | Array numerics |
| matplotlib | 3.4+ | Plotting |
| OpenMPI / MPICH | any | Parallel LAMMPS |

### Installing LAMMPS

```bash
sudo apt update
sudo apt install -y build-essential cmake openmpi-bin libopenmpi-dev libfftw3-dev

git clone --depth 1 https://github.com/lammps/lammps.git ~/lammps
cd ~/lammps && mkdir build && cd build

cmake ../cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=yes \
  -D PKG_KSPACE=yes \
  -D PKG_MOLECULE=yes \
  -D PKG_RIGID=yes

make -j$(nproc)
echo 'export PATH="$HOME/lammps/build:$PATH"' >> ~/.bashrc
source ~/.bashrc
lmp_mpi -h | head -3
```

**Required LAMMPS packages:** `KSPACE` (PPPM electrostatics), `MOLECULE`
(full atom style, bonds/angles), `RIGID` (SHAKE constraints).

### Installing PACKMOL

```bash
# Ubuntu 22+
sudo apt install packmol

# From source
wget https://m3g.github.io/packmol/packmol.tar.gz
tar -xzf packmol.tar.gz && cd packmol && make
sudo cp packmol /usr/local/bin/
```

### Python dependencies

```bash
pip install numpy matplotlib --break-system-packages
```

### Clone this repository

```bash
git clone https://github.com/<your-username>/water-methanol-interface.git
cd water-methanol-interface
```

---

## 5. Quick Start

Run a complete simulation with 10 methanol molecules in ~3 commands:

```bash
# Clone and enter repo
git clone https://github.com/<your-username>/water-methanol-interface.git
cd water-methanol-interface

# Make runner executable
chmod +x scripts/setup/run_simulation.sh

# Run everything: PACKMOL → LAMMPS → analysis
# Args: n_methanol (default 10), n_cores (default 4)
./scripts/setup/run_simulation.sh 10 4
```

This creates `simulations/sim_M10/` with all outputs inside it.

---

## 6. Step-by-Step Workflow

### Step 1 — Generate PACKMOL input

```bash
python3 scripts/setup/generate_packmol_input.py <n_methanol>
```

**What it does:** Writes `pack.inp` placing 16,800 water molecules in the slab
(z = 30–170 Å) and methanol molecules split equally between the bottom and top
vapour regions (z = 0–30 Å and z = 170–200 Å).

**Examples:**

```bash
python3 generate_packmol_input.py 10    # 5 methanol bottom + 5 top
python3 generate_packmol_input.py 20    # 10 + 10
python3 generate_packmol_input.py 50    # 25 + 25
```

> **Why split?** Placing all methanol in one vapour region (as the original
> `pack.inp` did) creates a non-symmetric initial condition. One interface
> starts with methanol present and the other bare. Symmetric splitting lets
> both interfaces equilibrate simultaneously and halves the effective
> equilibration time.

### Step 2 — Build initial structure

```bash
cd simulations/sim_M10
packmol < pack.inp > packmol.log 2>&1
```

Produces `system.xyz`. Check `packmol.log` for `Packing solved`. If PACKMOL
fails, try `tolerance 2.5` in place of `tolerance 2.0`.

### Step 3 — Convert to LAMMPS format

```bash
python3 xyz2lammps.py
```

Produces `system.data` with complete topology. The script reads molecule
counts from `pack.inp` dynamically — no editing needed if you changed
`n_methanol`.

Verify the output:

```bash
head -20 system.data
```

Expected:

```
LAMMPS data file – water/methanol liquid-vapour interface

50480 atoms        ← 16800*3 + 10*6 = 50,460 for M10
33600 bonds
16800 angles
...
6 atom types
```

### Step 4 — Run LAMMPS

```bash
mpirun -np 4 lmp_mpi -in scripts/setup/in.lammps -log simulation.log
```

**Estimated wall-clock time:** 24–48 h on 4 cores for the full 5 ns run.

For a cluster batch job (SLURM example):

```bash
#!/bin/bash
#SBATCH --ntasks=16 --time=48:00:00
module load lammps openmpi
mpirun lmp_mpi -in in.lammps -log simulation.log
```

### Step 5 — Diagnose the trajectory

```bash
python3 scripts/utils/check_dump.py --dump traj.lammpstrj --ncheck 2
```

Key checks:

| Check | Pass | Fail action |
|-------|------|-------------|
| mol column present | ✓ | Re-run with corrected in.lammps |
| Required columns | ✓ | — |
| Methanol mol grouping | ✓ non-zero | Mol column issue |
| Water in slab region | ~16800 atoms | Check PACKMOL setup |

### Step 6 — Run analysis

**General analysis (density + orientation + surface tension):**

```bash
cd simulations/sim_M10
python3 ../../scripts/analysis/analysis.py \
    --dump traj.lammpstrj \
    --log  simulation.log \
    --nmethanol 10
```

**Methanol-focused analysis:**

```bash
python3 ../../scripts/analysis/final_analysis_methanol.py \
    --dump traj.lammpstrj \
    --log  simulation.log \
    --nmethanol 10
```

Results appear in `analysis_results/` and `final_analysis_methanol/` inside the
simulation directory.

---

## 7. System Definition

```
z = 200 Å  ┌─────────────────────────┐
            │  Vapour  │ Methanol (top half)      │  z = 170–200 Å
z = 170 Å  ├ ─ ─ ─ ─ ─ GDS (top) ─ ─ ─ ─ ─ ─ ─ ┤
            │                                     │
            │          Water slab                 │  z = 30–170 Å
            │         (16,800 H₂O)                │
            │                                     │
z =  30 Å  ├ ─ ─ ─ ─ GDS (bottom) ─ ─ ─ ─ ─ ─ ─ ┤
            │  Vapour  │ Methanol (bottom half)   │  z = 0–30 Å
z =   0 Å  └─────────────────────────────────────┘
                    x, y: 60 Å (periodic)
```

| Property | Value |
|----------|-------|
| Box dimensions | 60 × 60 × 200 Å |
| Water molecules | 16,800 |
| Methanol molecules | 10 (default; parametric) |
| Total atoms (M10) | 50,460 |
| Temperature | 300 K |
| Boundary | Periodic in all directions |

---

## 8. Force Fields

### Water — SPC/Fw (flexible)

Wu, Y. et al. *J. Chem. Phys.* **124**, 024503 (2006)

| Parameter | Value |
|-----------|-------|
| q(O) | −0.82 e |
| q(H) | +0.41 e |
| ε(O–O) | 0.1554 kcal/mol |
| σ(O–O) | 3.166 Å |
| r₀(O–H) | 1.012 Å |
| k_bond | 4431.5 kcal/mol/Å² |
| θ₀(H–O–H) | 113.24° |
| k_angle | 55.0 kcal/mol/rad² |

O–H bonds and H–O–H angle held rigid by SHAKE (tol = 10⁻⁴ Å).

### Methanol — OPLS-AA

Jorgensen, W. L. et al. *J. Am. Chem. Soc.* **118**, 11225 (1996)

| Site | Type | q (e) | σ (Å) | ε (kcal/mol) |
|------|------|--------|--------|--------------|
| C (methyl) | CT | −0.18 | 3.50 | 0.0660 |
| O (hydroxyl) | OH | −0.683 | 3.12 | 0.1700 |
| H (methyl) | HC | +0.145 | 2.50 | 0.0300 |
| H (hydroxyl) | HO | +0.418 | 0.00 | 0.0000 |

Methanol O–H bond also constrained by SHAKE.

### Combining rules

Geometric mean (`pair_modify mix geometric`) for all cross LJ interactions.
Long-range electrostatics: PPPM (accuracy 10⁻⁴) with Yeh–Berkowitz slab
correction (`kspace_modify slab 3.0`). LJ cutoff: 10 Å.

---

## 9. Simulation Protocol

```
CG minimise  →  NVE 20 ps  →  NVT 480 ps (300 K)  →  NVE 5 ns (production)
```

| Setting | Value |
|---------|-------|
| Timestep | 1 fs |
| Thermostat | Nosé-Hoover, τ = 100 fs |
| Constraints | SHAKE: water O–H + H–O–H, methanol O–H |
| Trajectory dump | Every 1 ps (1000 steps) |
| Density profiles | Running average every 1000 steps |
| Production length | 5 ns (5,000,000 steps) |
| Analysis windows | 3–4 ns and 4–5 ns |

**Why NVE production?** After NVT equilibration the system is at thermal
equilibrium. Switching to NVE for production avoids thermostat perturbation of
interfacial fluctuations and gives correct surface tension from the pressure tensor.

---

## 10. Analysis Scripts Reference

### `scripts/utils/check_dump.py`

```
python3 check_dump.py [--dump FILE] [--ncheck N]
```

Reads N frames and reports column layout, atom counts, mol-ID grouping, and
z-distributions. Always run this before analysis.

---

### `scripts/analysis/analysis.py`

```
python3 analysis.py [--dump FILE] [--log FILE] [--nmethanol N]
                    [--outdir DIR] [--prod_start 500000]
                    [--nframes_max 0]
```

| Output file | Contents |
|-------------|----------|
| `surface_tension_blocks.pdf/png` | Bar chart: γ per 1 ns block |
| `surface_tension_blocks.txt` | Tabulated γ values |
| `density_number_5ns.pdf/png` | Water O, methanol C and O density profiles |
| `orientation_5ns.pdf/png` | Water ⟨cos θ⟩ at 4–5 ns |
| `orientation_4ns.pdf/png` | Water ⟨cos θ⟩ at 3–4 ns |
| `orientation_comparison.pdf/png` | Both windows overlaid |
| `profiles_5ns.csv` | All density + orientation data |

---

### `scripts/analysis/final_analysis_methanol.py`

```
python3 final_analysis_methanol.py [--dump FILE] [--log FILE]
                                   [--nmethanol N] [--outdir DIR]
                                   [--prod_start 500000]
```

| Output file | Contents |
|-------------|----------|
| `surface_tension_blocks.pdf/png` | Block-averaged surface tension |
| `methanol_number_density_5ns.pdf/png` | Methanol C and O number density |
| `methanol_mass_density_5ns.pdf/png` | Methanol mass density (g/cm³) |
| `methanol_oh_orientation_5ns.pdf/png` | Methanol O–H ⟨cos θ⟩ profile *(new)* |
| `water_orientation_5ns.pdf/png` | Water dipole ⟨cos θ⟩ at 4–5 ns |
| `orientation_summary.pdf/png` | Water + methanol orientations overlaid |
| `methanol_density_5ns.txt` | Density table |
| `all_profiles_5ns.csv` | All profiles in one file |

---

## 11. Output Files Guide

After a full run, `simulations/sim_M10/` contains:

```
sim_M10/
├── system.xyz                ← PACKMOL output
├── system.data               ← LAMMPS topology
├── traj.lammpstrj            ← Production trajectory (~10–20 GB)
├── traj.xyz                  ← XYZ trajectory (visualisation)
├── simulation.log            ← Thermo output
├── density_number.z.profile  ← LAMMPS-native running-average density
├── density_mass.z.profile
├── final.restart
├── final.data
│
├── analysis_results/
│   ├── surface_tension_blocks.pdf/.png/.txt
│   ├── density_number_5ns.pdf/.png
│   ├── orientation_5ns.pdf/.png
│   ├── orientation_4ns.pdf/.png
│   ├── orientation_comparison.pdf/.png
│   └── profiles_5ns.csv
│
└── final_analysis_methanol/
    ├── surface_tension_blocks.pdf/.png/.txt
    ├── methanol_number_density_5ns.pdf/.png
    ├── methanol_mass_density_5ns.pdf/.png
    ├── methanol_oh_orientation_5ns.pdf/.png
    ├── water_orientation_5ns.pdf/.png
    ├── orientation_summary.pdf/.png
    ├── methanol_density_5ns.txt
    └── all_profiles_5ns.csv
```

---

## 12. Known Issues & Fixes

### "ERROR: Lost atoms" during LAMMPS run

**Cause:** Steric clashes from PACKMOL not fully resolved by minimisation,
or timestep too large in early NVE.

**Fix:**
1. Ensure minimisation converged: look for `Energy initial ... final` in log
2. Increase PACKMOL tolerance to `2.5` if close contacts remain
3. Reduce timestep to `0.5` for the first NVE stage temporarily

### Surface tension is negative or very large

**Cause:** System not equilibrated; pzz fluctuations dominate early trajectory.

**Fix:** The analysis scripts only use frames from 3–5 ns. If the run is
shorter, adjust `--prod_start` and ensure `run 5000000` completed in the log.

### All methanol C atoms at z ≈ 185 Å (only top interface)

**Cause:** You are using the original `pack.inp` which placed all methanol in
the top region. Regenerate with:

```bash
python3 generate_packmol_input.py 10
```

### matplotlib backend error in headless environment

`UserWarning: Matplotlib is currently using agg` — this is **not an error**.
All scripts set `matplotlib.use("Agg")` explicitly to write PDF/PNG without
a display. Output files are still created correctly.

### Analysis finds 0 methanol molecules

**Cause:** `mol` column absent from dump (old trajectory from uncorrected `in.lammps`).

**Check:**
```bash
python3 scripts/utils/check_dump.py
```

If mol is absent, the analysis scripts fall back to mol = 0 for all atoms,
breaking molecule grouping. Re-run with the corrected `in.lammps` which includes
`mol` in the dump line.

---

## 13. Running Different Methanol Counts

The package is fully parametric. To simulate 20, 50, or 100 methanol molecules:

```bash
# 20 methanol
./scripts/setup/run_simulation.sh 20 4

# 50 methanol
./scripts/setup/run_simulation.sh 50 4

# 100 methanol
./scripts/setup/run_simulation.sh 100 4
```

Each run creates its own isolated directory (`simulations/sim_M20/`, etc.)
with all required files copied in. No manual editing needed.

---

## 14. Citation

If you use this simulation package, please cite:

```
[Author names]. "Water-Methanol Liquid-Vapour Interface MD Simulation Package."
Bangladesh University of Engineering and Technology (BUET), 2026.
```

**Force-field references:**

- SPC/Fw water: Wu, Y., Tepper, H. L. & Voth, G. A. *J. Chem. Phys.* **124**, 024503 (2006)
- OPLS-AA methanol: Jorgensen, W. L. et al. *J. Am. Chem. Soc.* **118**, 11225 (1996)
- LAMMPS: Thompson, A. P. et al. *Comput. Phys. Commun.* **271**, 108171 (2022)
- PACKMOL: Martínez, L. et al. *J. Comput. Chem.* **30**, 2157 (2009)
- Yeh–Berkowitz slab correction: Yeh, I.-C. & Berkowitz, M. L. *J. Chem. Phys.* **111**, 3155 (1999)

---

*Last updated: June 2026*
