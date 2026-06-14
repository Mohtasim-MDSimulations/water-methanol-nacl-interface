#!/bin/bash
# =============================================================================
# run_simulation.sh
# =============================================================================
# End-to-end runner: PACKMOL → LAMMPS → analysis.
#
# Usage:
#   chmod +x run_simulation.sh
#   ./run_simulation.sh [n_methanol] [n_cores]
#
#   Defaults: n_methanol=10, n_cores=4
#
# Estimated wall-clock time:
#   ~24–48 h on 4 cores for the full 5 ns production run.
# =============================================================================

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
N_METHANOL="${1:-10}"
N_CORES="${2:-4}"
LAMMPS_EXEC="lmp_mpi"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SETUP_DIR="${REPO_ROOT}/scripts/setup"
ANALYSIS_DIR="${REPO_ROOT}/scripts/analysis"
MOL_DIR="${REPO_ROOT}/molecule_files"

echo "========================================================"
echo "  Water–Methanol Interface Simulation"
echo "  Methanol molecules : ${N_METHANOL}"
echo "  MPI cores          : ${N_CORES}"
echo "  Repo root          : ${REPO_ROOT}"
echo "========================================================"
echo ""

# ── Prerequisite checks ───────────────────────────────────────────────────────
for cmd in python3 packmol mpirun "${LAMMPS_EXEC}"; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "ERROR: '${cmd}' not found. Install it and ensure it is on PATH."
        exit 1
    fi
    echo "  ✓  ${cmd}"
done
echo ""

# Required molecule files
for f in water.xyz methanol.xyz; do
    if [ ! -f "${MOL_DIR}/${f}" ]; then
        echo "ERROR: ${MOL_DIR}/${f} not found."
        exit 1
    fi
done

# ── Create working directory ──────────────────────────────────────────────────
SIM_DIR="${REPO_ROOT}/simulations/sim_M${N_METHANOL}"
mkdir -p "${SIM_DIR}"
cd "${SIM_DIR}"

# Copy inputs
cp "${MOL_DIR}"/{water.xyz,methanol.xyz} .
cp "${SETUP_DIR}"/{generate_packmol_input.py,xyz2lammps.py,in.lammps} .

echo "Working directory: ${SIM_DIR}"
echo ""

# ── Step 1: Generate PACKMOL input ────────────────────────────────────────────
echo "── Step 1: Generating PACKMOL input ────────────────────────"
python3 generate_packmol_input.py "${N_METHANOL}"

# ── Step 2: Run PACKMOL ───────────────────────────────────────────────────────
echo "── Step 2: Running PACKMOL ─────────────────────────────────"
packmol < pack.inp > packmol.log 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: PACKMOL failed. See packmol.log"
    exit 1
fi
echo "  ✓  PACKMOL complete"

# ── Step 3: Convert to LAMMPS data file ──────────────────────────────────────
echo "── Step 3: Converting XYZ → LAMMPS data file ───────────────"
python3 xyz2lammps.py
echo "  ✓  system.data written"

# ── Step 4: Run LAMMPS ───────────────────────────────────────────────────────
echo "── Step 4: Running LAMMPS (${N_CORES} cores, ~24–48 h) ─────"
mpirun -np "${N_CORES}" "${LAMMPS_EXEC}" -in in.lammps -log simulation.log
if [ $? -ne 0 ]; then
    echo "ERROR: LAMMPS failed. See simulation.log"
    exit 1
fi
echo "  ✓  LAMMPS simulation complete"

# ── Step 5: Analysis ─────────────────────────────────────────────────────────
echo "── Step 5: Running analysis ────────────────────────────────"
python3 "${ANALYSIS_DIR}/analysis.py" \
    --dump traj.lammpstrj \
    --log  simulation.log \
    --nmethanol "${N_METHANOL}"
echo "  ✓  General analysis complete"

python3 "${ANALYSIS_DIR}/final_analysis_methanol.py" \
    --dump traj.lammpstrj \
    --log  simulation.log \
    --nmethanol "${N_METHANOL}"
echo "  ✓  Methanol-focused analysis complete"

echo ""
echo "========================================================"
echo "  Workflow complete. Results in ${SIM_DIR}/"
echo "========================================================"
