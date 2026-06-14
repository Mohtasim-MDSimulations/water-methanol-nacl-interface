#!/bin/bash
# =============================================================================
# run_simulation.sh
# =============================================================================
# End-to-end runner: PACKMOL → data conversion → equilibration → production
#
# Usage:
#   chmod +x run_simulation.sh
#   ./run_simulation.sh [n_nacl_pairs] [n_cores]
#
#   Defaults: n_nacl=10, n_cores=4
#
# Estimated wall-clock time:
#   Equilibration : ~3–5 h on 4 cores
#   Production    : ~25–35 h on 4 cores
# =============================================================================

set -euo pipefail

N_NACL="${1:-10}"
N_CORES="${2:-4}"
LAMMPS_EXEC="lmp"      # adjust to lmp_mpi if needed

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SETUP_DIR="${REPO_ROOT}/scripts/setup"
ANALYSIS_DIR="${REPO_ROOT}/scripts/analysis"
MOL_DIR="${REPO_ROOT}/molecule_files"

echo "============================================================"
echo "  Water–NaCl Interface Simulation"
echo "  NaCl ion pairs : ${N_NACL}"
echo "  MPI cores      : ${N_CORES}"
echo "  Repo root      : ${REPO_ROOT}"
echo "============================================================"
echo ""

# Prerequisite check
for cmd in python3 packmol mpirun "${LAMMPS_EXEC}"; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "ERROR: '${cmd}' not found."
        exit 1
    fi
    echo "  ✓  ${cmd}"
done
echo ""

# Create simulation directory
SIM_DIR="${REPO_ROOT}/simulations/sim_N${N_NACL}"
mkdir -p "${SIM_DIR}"
cd "${SIM_DIR}"

# Copy molecule files and scripts
cp "${MOL_DIR}"/{water.pdb,na.pdb,cl.pdb} .
cp "${SETUP_DIR}"/{generate_packmol_input.py,pdb_to_lammps.py,equilibration.in,production.in} .

echo "Working directory: ${SIM_DIR}"
echo ""

# Step 1: Generate PACKMOL input
echo "── Step 1: Generating PACKMOL input ───────────────────────"
python3 generate_packmol_input.py "${N_NACL}"

# Step 2: Run PACKMOL
echo "── Step 2: Running PACKMOL ────────────────────────────────"
packmol < water_slab.inp > packmol.log 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: PACKMOL failed. Check packmol.log"
    exit 1
fi
echo "  ✓  PACKMOL complete"

# Step 3: Convert PDB to LAMMPS data file
echo "── Step 3: Converting PDB → LAMMPS data file ──────────────"
python3 pdb_to_lammps.py water_slab.pdb water_interface.data
echo "  ✓  water_interface.data written"

# Step 4: Equilibration
echo "── Step 4: Running equilibration (~4 h) ────────────────────"
mpirun -np "${N_CORES}" "${LAMMPS_EXEC}" -in equilibration.in \
    -log equilibration.log
if [ $? -ne 0 ]; then
    echo "ERROR: Equilibration failed. Check equilibration.log"
    exit 1
fi
echo "  ✓  Equilibration complete"

# Step 5: Production
echo "── Step 5: Running production (~30 h) ──────────────────────"
mpirun -np "${N_CORES}" "${LAMMPS_EXEC}" -in production.in \
    -log production.log
if [ $? -ne 0 ]; then
    echo "ERROR: Production failed. Check production.log"
    exit 1
fi
echo "  ✓  Production complete"

# Step 6: Analysis
echo "── Step 6: Running analysis ────────────────────────────────"
python3 "${ANALYSIS_DIR}/final_analysis_complete.py" \
    --traj  production.lammpstrj \
    --st    surface_tension.dat \
    --outdir "${REPO_ROOT}/results/sim_N${N_NACL}"
echo "  ✓  Analysis complete"

echo ""
echo "============================================================"
echo "  Workflow complete. Results in:"
echo "  ${REPO_ROOT}/results/sim_N${N_NACL}/"
echo "============================================================"
