#!/usr/bin/env bash
# =============================================================================
# build_apptainer.sh - Build SAM3D Apptainer/Singularity image (Terminal)
# =============================================================================
# This script builds the Apptainer image directly in the terminal.
# Use this if you don't have SLURM or want to build locally.
#
# Usage:
#   ./scripts/build_apptainer.sh
#
# Requirements:
#   - Apptainer/Singularity installed
#   - ~128GB RAM available
#   - Internet access to download base image and packages
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"
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
    exit 0
fi

# =============================================================================
# Set up Apptainer cache and temp directories
# =============================================================================
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
    2>&1 | tee "${PROJECT_DIR}/logs/apptainer_build.log"

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
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "BUILD FAILED!"
    echo "=============================================="
    echo "Check logs/apptainer_build.log for details"
    exit 1
fi
