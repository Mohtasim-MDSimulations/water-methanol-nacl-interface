#!/bin/bash
# =============================================================================
# run_all_simulations.sh
# =============================================================================
# Batch runner for all seven mixed and pure-component simulations:
#
#   Mixed systems:   M10N10, M20N20, M50N50, M100N100
#   Pure NaCl:       N10  (pure NaCl baseline, 10 ion pairs)
#   Pure methanol:   M10  (pure methanol baseline, 10 molecules)
#
# Each simulation is built in its own subdirectory:
#   sim_M10_N10/   sim_M20_N20/   sim_M50_N50/   sim_M100_N100/
#   sim_N10/       sim_M10/
#
# Prerequisites (must be on PATH):
#   python3, packmol, mpirun, lmp_mpi
#
# Usage:
#   chmod +x run_all_simulations.sh
#   ./run_all_simulations.sh
#
# Estimated wall-clock time per simulation: ~12–24 h on 6 cores (depends on CPU)
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_CORES=6
LAMMPS_EXEC="lmp_mpi"
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root
SETUP_DIR="$(cd "$(dirname "$0")" && pwd)"        # scripts/setup/
MOL_DIR="${BASE_DIR}/molecule_files"

# ── Prerequisite check ────────────────────────────────────────────────────────
echo "============================================================"
echo "  Checking prerequisites..."
echo "============================================================"
for cmd in python3 packmol mpirun "${LAMMPS_EXEC}"; do
    if ! command -v "${cmd}" &> /dev/null; then
        echo "ERROR: '${cmd}' not found. Please install it and ensure it is on PATH."
        exit 1
    fi
    echo "  ✓  ${cmd}"
done

required_mol_files=("water.xyz" "methanol.xyz" "na.xyz" "cl.xyz")
for f in "${required_mol_files[@]}"; do
    if [ ! -f "${MOL_DIR}/${f}" ]; then
        echo "ERROR: molecule file ${MOL_DIR}/${f} not found."
        exit 1
    fi
done
echo ""

# ── Helper: build one simulation directory ────────────────────────────────────
run_simulation() {
    local sim_name="$1"
    local n_methanol="$2"
    local n_nacl="$3"

    local SIM_DIR="${BASE_DIR}/simulations/${sim_name}"
    echo "------------------------------------------------------------"
    echo "  Building simulation: ${sim_name}  (methanol=${n_methanol}, NaCl=${n_nacl})"
    echo "------------------------------------------------------------"

    mkdir -p "${SIM_DIR}"
    cd "${SIM_DIR}"

    # Copy molecule files and scripts
    cp "${MOL_DIR}"/{water.xyz,methanol.xyz,na.xyz,cl.xyz} .
    cp "${SETUP_DIR}"/{generate_packmol_input.py,xyz2lammps.py,in.lammps} .

    # Generate PACKMOL input
    python3 generate_packmol_input.py "${n_methanol}" "${n_nacl}"

    # Run PACKMOL to generate initial coordinates
    echo "  Running PACKMOL..."
    packmol < pack.inp > packmol.log 2>&1
    if [ $? -ne 0 ]; then
        echo "  ERROR: PACKMOL failed. Check packmol.log"
        exit 1
    fi

    # Convert to LAMMPS data file
    echo "  Converting XYZ → LAMMPS data file..."
    python3 xyz2lammps.py

    # Run LAMMPS simulation
    echo "  Running LAMMPS (${NUM_CORES} cores)..."
    mpirun -np "${NUM_CORES}" "${LAMMPS_EXEC}" -in in.lammps -log simulation.log

    echo "  ✓  ${sim_name} complete."
    cd "${BASE_DIR}"
}

# ── Define all seven systems ──────────────────────────────────────────────────
echo "============================================================"
echo "  Starting all simulations"
echo "============================================================"
echo ""

# Pure baselines
run_simulation "sim_N10"       0   10    # Pure NaCl baseline
run_simulation "sim_M10"      10    0    # Pure methanol baseline

# Mixed ternary systems
run_simulation "sim_M10_N10"  10   10
run_simulation "sim_M20_N20"  20   20
run_simulation "sim_M50_N50"  50   50
run_simulation "sim_M100_N100" 100 100

echo ""
echo "============================================================"
echo "  All simulations finished successfully."
echo "  Results are in: ${BASE_DIR}/simulations/"
echo "============================================================"
