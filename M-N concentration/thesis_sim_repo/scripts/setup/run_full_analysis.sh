#!/bin/bash
# =============================================================================
# run_full_analysis.sh
# =============================================================================
# Master script: runs all post-simulation analysis in the correct order.
#
# Assumes:
#   - All simulation directories exist under simulations/
#   - traj.lammpstrj and system.data are present in each
#   - Run from the repository root
#
# Steps:
#   1. check_dump.py       – sanity check on each trajectory
#   2. analysis_fixed.py   – per-system density + orientation plots
#   3. analyze_all_fixed.py – multi-concentration comparison plots
#   4. surface_excess.py   – block-averaged Γ for each mixed system
#   5. hbond_profile.py    – H-bond profile across all systems
#
# Usage:
#   chmod +x scripts/setup/run_full_analysis.sh
#   ./scripts/setup/run_full_analysis.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ANALYSIS_DIR="${REPO_ROOT}/scripts/analysis"
UTILS_DIR="${REPO_ROOT}/scripts/utils"
SIM_BASE="${REPO_ROOT}/simulations"
RESULTS_DIR="${REPO_ROOT}/results"

mkdir -p "${RESULTS_DIR}"

# All simulation subdirectory names
SIM_NAMES=(sim_N10 sim_M10 sim_M10_N10 sim_M20_N20 sim_M50_N50 sim_M100_N100)

echo "============================================================"
echo "  Full analysis pipeline"
echo "  Repo root: ${REPO_ROOT}"
echo "============================================================"
echo ""

# ── Step 1: Trajectory diagnostics ───────────────────────────────────────────
echo "── Step 1: check_dump.py ───────────────────────────────────"
for sim in "${SIM_NAMES[@]}"; do
    SPATH="${SIM_BASE}/${sim}"
    [ -d "${SPATH}" ] || continue
    echo "  Checking ${sim} ..."
    cd "${SPATH}"
    python3 "${UTILS_DIR}/check_dump.py" --ncheck 1 2>&1 | tail -20
    cd "${REPO_ROOT}"
done
echo ""

# ── Step 2: Per-system analysis ───────────────────────────────────────────────
echo "── Step 2: analysis_fixed.py (per system) ──────────────────"
for sim in "${SIM_NAMES[@]}"; do
    SPATH="${SIM_BASE}/${sim}"
    [ -d "${SPATH}" ] || continue
    echo "  Analysing ${sim} ..."
    cd "${SPATH}"
    python3 "${ANALYSIS_DIR}/analysis_fixed.py" \
        --nframes 50 --binwidth 1.0 \
        --outdir "${RESULTS_DIR}/per_system"
    cd "${REPO_ROOT}"
done
echo ""

# ── Step 3: Multi-concentration comparison ────────────────────────────────────
echo "── Step 3: analyze_all_fixed.py (comparisons) ──────────────"
cd "${REPO_ROOT}"
python3 "${ANALYSIS_DIR}/analyze_all_fixed.py" \
    --nframes 50 --binwidth 1.0 \
    --outdir "${RESULTS_DIR}/line_graph_fixed"
echo ""

# ── Step 4: Surface excess for mixed systems ──────────────────────────────────
echo "── Step 4: surface_excess.py ───────────────────────────────"
for sim in sim_M10_N10 sim_M20_N20 sim_M50_N50 sim_M100_N100; do
    SPATH="${SIM_BASE}/${sim}"
    [ -d "${SPATH}" ] || continue
    echo "  Surface excess: ${sim} ..."
    cd "${SPATH}"
    python3 "${ANALYSIS_DIR}/surface_excess.py"
    cd "${REPO_ROOT}"
done
echo ""

# ── Step 5: H-bond profiles ───────────────────────────────────────────────────
echo "── Step 5: hbond_profile.py ────────────────────────────────"
cd "${REPO_ROOT}"
python3 "${ANALYSIS_DIR}/hbond_profile.py" \
    --run_all --nframes 50
echo ""

echo "============================================================"
echo "  Analysis complete. Results in: ${RESULTS_DIR}/"
echo "============================================================"
