#!/usr/bin/env bash
# =============================================================================
# run_webapp_apptainer.sh - Run SAM3D webapp using Apptainer (Terminal)
# =============================================================================
# This script runs the webapp inside an Apptainer container directly.
# Use this if you don't have SLURM or want to run locally.
#
# Usage:
#   ./scripts/run_webapp_apptainer.sh
#
# Access:
#   Open http://localhost:8000 in your browser
#
# Requirements:
#   - NVIDIA GPU with 32GB+ VRAM
#   - NVIDIA Driver 560+ with CUDA support
#   - Apptainer with --nv (NVIDIA) support
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"

PORT="${PORT:-8000}"
SIF_NAME="${SIF_NAME:-sam3d.sif}"
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
echo "SAM3D Web Application - Apptainer"
echo "=============================================="
echo "Project Dir: ${PROJECT_DIR}"
echo "SIF Image:   ${PROJECT_DIR}/${SIF_NAME}"
echo "Port:        ${PORT}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "=============================================="

# =============================================================================
# Check Apptainer Image
# =============================================================================
if [[ ! -f "${PROJECT_DIR}/${SIF_NAME}" ]]; then
    echo "ERROR: Apptainer image '${SIF_NAME}' not found!"
    echo "Please build it first with:"
    echo "  ./scripts/build_apptainer.sh"
    exit 1
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
# Run Apptainer Container
# =============================================================================
echo "Starting Apptainer container..."

# Create writable directories for JIT compilation caches
NVDIFFRAST_TMP="${PROJECT_DIR}/.nvdiffrast_cache"
TORCH_EXTENSIONS="${PROJECT_DIR}/.torch_extensions"
mkdir -p "${NVDIFFRAST_TMP}" "${TORCH_EXTENSIONS}"

apptainer exec \
    --nv \
    --cleanenv \
    --no-home \
    --writable-tmpfs \
    --bind "${PROJECT_DIR}/sam3:/app/sam3:ro" \
    --bind "${PROJECT_DIR}/sam-3d-objects:/app/sam-3d-objects:ro" \
    --bind "${PROJECT_DIR}/lib:/app/lib:ro" \
    --bind "${PROJECT_DIR}/pipeline.py:/app/pipeline.py:ro" \
    --bind "${PROJECT_DIR}/webapp:/app/webapp:ro" \
    --bind "${OUTPUT_DIR}:/app/webapp_outputs" \
    --bind "${CHECKPOINTS_DIR}:/app/sam-3d-objects/checkpoints:ro" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --bind "${TORCH_CACHE}:/root/.cache/torch" \
    --bind "${NVDIFFRAST_TMP}:/tmp/nvdiffrast_cache" \
    --bind "${TORCH_EXTENSIONS}:/root/.cache/torch_extensions" \
    --env PORT="${PORT}" \
    --env PYTHONUNBUFFERED=1 \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    --env NVDIFFRAST_CACHE_DIR="/tmp/nvdiffrast_cache" \
    --env HYDRA_FULL_ERROR=1 \
    --env CONDA_PREFIX="/usr/local/cuda" \
    --env MPLCONFIGDIR="/tmp/matplotlib" \
    --env SPARSE_ATTN_BACKEND=sdpa \
    --env ATTN_BACKEND=sdpa \
    --env SSL_CERT_FILE="" \
    --env HF_HOME="/root/.cache/huggingface" \
    --env HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface/hub" \
    --env TORCH_HOME="/root/.cache/torch" \
    --env TORCH_EXTENSIONS_DIR="/root/.cache/torch_extensions" \
    --env HF_TOKEN="$(cat ${HF_CACHE}/token 2>/dev/null || echo '')" \
    --pwd /app \
    "${PROJECT_DIR}/${SIF_NAME}" \
    bash -c "pip install --quiet plyfile && python -u /app/webapp/app.py"

echo ""
echo "Container exited."
