#!/usr/bin/env bash
# =============================================================================
# build_docker.sh - Build SAM3D Docker image
# =============================================================================
# This script builds the Docker image.
#
# Usage:
#   ./scripts/build_docker.sh
#
# Requirements:
#   - Docker with GPU support (nvidia-docker2 or nvidia-container-toolkit)
#   - Internet access to download base image and packages
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"
IMAGE_NAME="${IMAGE_NAME:-sam3d}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FORCE_BUILD="${FORCE_BUILD:-false}"

# =============================================================================
# Setup
# =============================================================================
cd "${PROJECT_DIR}"
mkdir -p logs

echo "=============================================="
echo "SAM3D Docker Image Builder"
echo "=============================================="
echo "Project Dir:  ${PROJECT_DIR}"
echo "Image:        ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=============================================="

# =============================================================================
# Check if image already exists
# =============================================================================
if docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null && [[ "${FORCE_BUILD}" != "true" ]]; then
    echo ""
    echo "Image ${IMAGE_NAME}:${IMAGE_TAG} already exists!"
    echo "Use FORCE_BUILD=true to rebuild."
    exit 0
fi

# =============================================================================
# Build Docker Image
# =============================================================================
echo ""
echo "Building Docker image..."
echo "This may take 30-60 minutes on first build."
echo ""

BUILD_START=$(date +%s)

docker build \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --file "${PROJECT_DIR}/docker/Dockerfile" \
    "${PROJECT_DIR}" \
    2>&1 | tee "${PROJECT_DIR}/logs/docker_build.log"

BUILD_STATUS=$?
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

if [[ ${BUILD_STATUS} -eq 0 ]]; then
    echo ""
    echo "=============================================="
    echo "BUILD SUCCESSFUL!"
    echo "=============================================="
    echo "Image:      ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Build time: ${BUILD_TIME} seconds"
    echo ""
    echo "Image details:"
    docker image ls "${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "BUILD FAILED!"
    echo "=============================================="
    echo "Check logs/docker_build.log for details"
    exit 1
fi
