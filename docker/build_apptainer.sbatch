#!/usr/bin/env bash
# =============================================================================
# build_apptainer.sbatch - Slurm job to build SAM3D Apptainer/Singularity image
# =============================================================================
# This script builds the Apptainer image on a compute node.
#
# Usage:
#   sbatch docker/build_apptainer.sbatch
# =============================================================================

#SBATCH -J sam3d_build
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
# CUSTOMIZE: Uncomment and set your SLURM partition/QOS as needed
# #SBATCH --partition=gpu
# #SBATCH --qos=your_qos_here
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
# PROJECT_DIR: Set this to your project directory, or pass via environment
# Example: PROJECT_DIR=/path/to/sam3d-pipeline sbatch docker/build_apptainer.sbatch
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SIF_NAME="${SIF_NAME:-sam3d.sif}"
FORCE_BUILD="${FORCE_BUILD:-false}"

# =============================================================================
# Setup
# =============================================================================
cd "${PROJECT_DIR}"
mkdir -p logs

echo "=============================================="
echo "SAM3D Apptainer Image Builder"
echo "=============================================="
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Node:         ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "Project Dir:  ${PROJECT_DIR}"
echo "SIF Name:     ${SIF_NAME}"
echo "=============================================="

# =============================================================================
# Check if image already exists
# =============================================================================
if [[ -f "${PROJECT_DIR}/${SIF_NAME}" ]] && [[ "${FORCE_BUILD}" != "true" ]]; then
    echo ""
    echo "Image ${SIF_NAME} already exists!"
    echo "Use FORCE_BUILD=true to rebuild."
    echo "SUCCESS" > "${PROJECT_DIR}/logs/build_status_${SLURM_JOB_ID:-manual}.txt"
    echo "Image: ${SIF_NAME} (existing)" >> "${PROJECT_DIR}/logs/build_status_${SLURM_JOB_ID:-manual}.txt"
    exit 0
fi

# =============================================================================
# Set up Apptainer cache and temp directories
# =============================================================================
# Use local scratch or project directory for cache
export APPTAINER_CACHEDIR="${PROJECT_DIR}/.apptainer_cache"
export APPTAINER_TMPDIR="${PROJECT_DIR}/.apptainer_tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo ""
echo "Cache dir: ${APPTAINER_CACHEDIR}"
echo "Temp dir:  ${APPTAINER_TMPDIR}"

# =============================================================================
# Build Apptainer Image
# =============================================================================
echo ""
echo "Building Apptainer image..."
echo "This may take 30-60 minutes on first build."
echo ""

BUILD_START=$(date +%s)

# Remove old image if force rebuilding
if [[ "${FORCE_BUILD}" == "true" ]] && [[ -f "${PROJECT_DIR}/${SIF_NAME}" ]]; then
    rm -f "${PROJECT_DIR}/${SIF_NAME}"
fi

# Build the image
apptainer build \
    "${PROJECT_DIR}/${SIF_NAME}" \
    "${PROJECT_DIR}/docker/sam3d.def" \
    2>&1 | tee "${PROJECT_DIR}/logs/apptainer_build_${SLURM_JOB_ID:-manual}.log"

BUILD_STATUS=$?
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

if [[ ${BUILD_STATUS} -eq 0 ]] && [[ -f "${PROJECT_DIR}/${SIF_NAME}" ]]; then
    echo ""
    echo "=============================================="
    echo "BUILD SUCCESSFUL!"
    echo "=============================================="
    echo "Image:      ${PROJECT_DIR}/${SIF_NAME}"
    echo "Build time: ${BUILD_TIME} seconds"
    echo ""
    echo "Image size:"
    ls -lh "${PROJECT_DIR}/${SIF_NAME}"
    echo ""

    # Write success marker
    echo "SUCCESS" > "${PROJECT_DIR}/logs/build_status_${SLURM_JOB_ID:-manual}.txt"
    echo "Image: ${SIF_NAME}" >> "${PROJECT_DIR}/logs/build_status_${SLURM_JOB_ID:-manual}.txt"
    echo "Build time: ${BUILD_TIME}s" >> "${PROJECT_DIR}/logs/build_status_${SLURM_JOB_ID:-manual}.txt"
    echo "Size: $(ls -lh "${PROJECT_DIR}/${SIF_NAME}" | awk '{print $5}')" >> "${PROJECT_DIR}/logs/build_status_${SLURM_JOB_ID:-manual}.txt"

    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "BUILD FAILED!"
    echo "=============================================="
    echo "Check logs/apptainer_build_${SLURM_JOB_ID:-manual}.log for details"

    # Write failure marker
    echo "FAILED" > "${PROJECT_DIR}/logs/build_status_${SLURM_JOB_ID:-manual}.txt"

    exit 1
fi
