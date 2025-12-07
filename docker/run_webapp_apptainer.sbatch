#!/usr/bin/env bash
# =============================================================================
# run_webapp_apptainer.sbatch - Run SAM3D webapp using Apptainer
# =============================================================================
# This script runs the webapp inside an Apptainer container on a GPU node.
#
# Usage:
#   sbatch docker/run_webapp_apptainer.sbatch
#
# Access (from outside the cluster):
#   1. SSH tunnel: ssh -L 8000:<node>:8000 <user>@<YOUR_CLUSTER_LOGIN_NODE>
#   2. Open: http://localhost:8000
# =============================================================================

#SBATCH -J sam3d_webapp
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# CUSTOMIZE: Set GPU type based on your cluster (A100, H100, H200, etc.)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
# CUSTOMIZE: Uncomment and set your SLURM partition/QOS as needed
# #SBATCH --partition=gpu
# #SBATCH --qos=your_qos_here
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
PORT="${PORT:-8000}"
# PROJECT_DIR: Set this to your project directory, or pass via environment
# Example: PROJECT_DIR=/path/to/sam3d-pipeline sbatch docker/run_webapp_apptainer.sbatch
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SIF_NAME="${SIF_NAME:-sam3d.sif}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/webapp_outputs}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_DIR}/sam-3d-objects/checkpoints}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-${HOME}/.cache/torch}"

# CUSTOMIZE: Set your cluster's login node for SSH tunnel instructions
# Example: LOGIN_NODE="login.mycluster.edu"
LOGIN_NODE="${LOGIN_NODE:-<YOUR_CLUSTER_LOGIN_NODE>}"

# =============================================================================
# Setup
# =============================================================================
mkdir -p logs "${OUTPUT_DIR}"

NODE_NAME=$(hostname)
NODE_IP=$(hostname -i | awk '{print $1}')

echo "=============================================="
echo "SAM3D Web Application - Apptainer"
echo "=============================================="
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Node:        ${NODE_NAME}"
echo "Node IP:     ${NODE_IP}"
echo "GPUs:        ${SLURM_GPUS_ON_NODE:-N/A}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Memory:      ${SLURM_MEM_PER_NODE:-N/A} MB"
echo "=============================================="
echo "SIF Image:   ${PROJECT_DIR}/${SIF_NAME}"
echo "Port:        ${PORT}"
echo "Project Dir: ${PROJECT_DIR}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "=============================================="

# =============================================================================
# Check Apptainer Image
# =============================================================================
if [[ ! -f "${PROJECT_DIR}/${SIF_NAME}" ]]; then
    echo "ERROR: Apptainer image '${SIF_NAME}' not found!"
    echo "Please build it first with:"
    echo "  sbatch docker/build_apptainer.sbatch"
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
echo "Option 1: SSH Tunnel (Recommended for external access)"
echo "  Step 1: Open a NEW terminal and run:"
echo "    ssh -L ${PORT}:${NODE_NAME}:${PORT} ${USER}@${LOGIN_NODE}"
echo ""
echo "  Step 2: Open in your browser:"
echo "    http://localhost:${PORT}"
echo ""
echo "Option 2: Direct Access (only from within cluster network)"
echo "  http://${NODE_IP}:${PORT}"
echo "  http://${NODE_NAME}:${PORT}"
echo ""
echo "=============================================="
echo ""

# Save access info to file
cat > "${PROJECT_DIR}/logs/webapp_url_${SLURM_JOB_ID:-manual}.txt" << EOF
SAM3D Webapp Access Information
===============================
Job ID: ${SLURM_JOB_ID:-N/A}
Node:   ${NODE_NAME}
Port:   ${PORT}

SSH Tunnel Command (run in a new terminal):
  ssh -L ${PORT}:${NODE_NAME}:${PORT} ${USER}@${LOGIN_NODE}

Then open: http://localhost:${PORT}

Direct URLs (cluster network only):
  http://${NODE_IP}:${PORT}
  http://${NODE_NAME}:${PORT}
EOF

# =============================================================================
# Run Apptainer Container
# =============================================================================
echo "Starting Apptainer container..."

# Set environment for nvdiffrast JIT compilation
# CUSTOMIZE: Adjust TORCH_CUDA_ARCH_LIST for your GPU architecture
# Common values: 8.0 (A100), 8.6 (RTX 3090), 8.9 (RTX 4090), 9.0 (H100/H200)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

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
    --env CUDA_VISIBLE_DEVICES=0 \
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
