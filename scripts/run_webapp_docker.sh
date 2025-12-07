#!/usr/bin/env bash
# =============================================================================
# run_webapp_docker.sh - Run SAM3D webapp using Docker
# =============================================================================
# This script runs the webapp inside a Docker container.
#
# Usage:
#   ./scripts/run_webapp_docker.sh
#
# Access:
#   Open http://localhost:8000 in your browser
#
# Requirements:
#   - NVIDIA GPU with 32GB+ VRAM
#   - Docker with GPU support (nvidia-docker2 or nvidia-container-toolkit)
#   - NVIDIA Driver 560+ with CUDA support
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"

PORT="${PORT:-8000}"
IMAGE_NAME="${IMAGE_NAME:-sam3d}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-sam3d_webapp}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/webapp_outputs}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_DIR}/sam-3d-objects/checkpoints}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-${HOME}/.cache/torch}"

# GPU architecture for JIT compilation
# Common values: 8.0 (A100), 8.6 (RTX 3090), 8.9 (RTX 4090), 9.0 (H100/H200)
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

# =============================================================================
# Setup
# =============================================================================
cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=============================================="
echo "SAM3D Web Application - Docker"
echo "=============================================="
echo "Project Dir: ${PROJECT_DIR}"
echo "Image:       ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Container:   ${CONTAINER_NAME}"
echo "Port:        ${PORT}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "=============================================="

# =============================================================================
# Check Docker Image
# =============================================================================
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    echo "ERROR: Docker image '${IMAGE_NAME}:${IMAGE_TAG}' not found!"
    echo "Please build it first with:"
    echo "  ./scripts/build_docker.sh"
    exit 1
fi

# =============================================================================
# Stop existing container if running
# =============================================================================
if docker container inspect "${CONTAINER_NAME}" &> /dev/null; then
    echo "Stopping existing container '${CONTAINER_NAME}'..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# =============================================================================
# Display Access Instructions
# =============================================================================
echo ""
echo "=============================================="
echo "ACCESS THE WEBAPP"
echo "=============================================="
echo ""
echo "Open in your browser:"
echo "  http://localhost:${PORT}"
echo ""
echo "=============================================="
echo ""

# =============================================================================
# Create writable directories for caches
# =============================================================================
NVDIFFRAST_TMP="${PROJECT_DIR}/.nvdiffrast_cache"
TORCH_EXTENSIONS="${PROJECT_DIR}/.torch_extensions"
mkdir -p "${NVDIFFRAST_TMP}" "${TORCH_EXTENSIONS}"

# =============================================================================
# Run Docker Container
# =============================================================================
echo "Starting Docker container..."

docker run \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --rm \
    -it \
    -p "${PORT}:${PORT}" \
    -v "${PROJECT_DIR}/sam3:/app/sam3:ro" \
    -v "${PROJECT_DIR}/sam-3d-objects:/app/sam-3d-objects:ro" \
    -v "${PROJECT_DIR}/lib:/app/lib:ro" \
    -v "${PROJECT_DIR}/pipeline.py:/app/pipeline.py:ro" \
    -v "${PROJECT_DIR}/webapp:/app/webapp:ro" \
    -v "${OUTPUT_DIR}:/app/webapp_outputs" \
    -v "${CHECKPOINTS_DIR}:/app/sam-3d-objects/checkpoints:ro" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${TORCH_CACHE}:/root/.cache/torch" \
    -v "${NVDIFFRAST_TMP}:/tmp/nvdiffrast_cache" \
    -v "${TORCH_EXTENSIONS}:/root/.cache/torch_extensions" \
    -e PORT="${PORT}" \
    -e PYTHONUNBUFFERED=1 \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    -e NVDIFFRAST_CACHE_DIR="/tmp/nvdiffrast_cache" \
    -e HYDRA_FULL_ERROR=1 \
    -e CONDA_PREFIX="/usr/local/cuda" \
    -e MPLCONFIGDIR="/tmp/matplotlib" \
    -e SPARSE_ATTN_BACKEND=sdpa \
    -e ATTN_BACKEND=sdpa \
    -e SSL_CERT_FILE="" \
    -e HF_HOME="/root/.cache/huggingface" \
    -e HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface/hub" \
    -e TORCH_HOME="/root/.cache/torch" \
    -e TORCH_EXTENSIONS_DIR="/root/.cache/torch_extensions" \
    -e HF_TOKEN="$(cat ${HF_CACHE}/token 2>/dev/null || echo '')" \
    -w /app \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash -c "pip install --quiet plyfile && python -u /app/webapp/app.py"

echo ""
echo "Container exited."
