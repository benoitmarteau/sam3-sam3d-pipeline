#!/usr/bin/env bash
# =============================================================================
# setup.sh - Initial setup script for SAM3D Pipeline
# =============================================================================
# This script clones the required META repositories and creates directories.
#
# Usage:
#   ./scripts/setup.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=============================================="
echo "SAM3D Pipeline Setup"
echo "=============================================="
echo "Project directory: ${PROJECT_DIR}"
echo ""

cd "${PROJECT_DIR}"

# =============================================================================
# Clone META repositories
# =============================================================================

if [[ ! -d "sam3" ]]; then
    echo ">>> Cloning SAM3 (Segment Anything Model 3)..."
    git clone https://github.com/facebookresearch/sam3.git
else
    echo ">>> sam3/ already exists, skipping clone"
fi

if [[ ! -d "sam-3d-objects" ]]; then
    echo ">>> Cloning SAM3D (3D Object Reconstruction)..."
    git clone https://github.com/facebookresearch/sam-3d-objects.git
else
    echo ">>> sam-3d-objects/ already exists, skipping clone"
fi

# =============================================================================
# Create required directories
# =============================================================================

echo ""
echo ">>> Creating required directories..."
mkdir -p logs
mkdir -p webapp_outputs

# =============================================================================
# Check for checkpoints
# =============================================================================

echo ""
if [[ -d "sam-3d-objects/checkpoints/hf" ]]; then
    echo ">>> SAM3D checkpoints found!"
else
    echo ">>> WARNING: SAM3D checkpoints not found!"
    echo "    Please download checkpoints to: sam-3d-objects/checkpoints/hf/"
    echo "    See: https://github.com/facebookresearch/sam-3d-objects#setup"
fi

# =============================================================================
# Check for HuggingFace authentication
# =============================================================================

echo ""
if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    echo ">>> HuggingFace token found!"
else
    echo ">>> WARNING: HuggingFace token not found!"
    echo "    Please run: huggingface-cli login"
    echo "    You also need access to: https://huggingface.co/facebook/sam3"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Download SAM3D checkpoints (if not already done)"
echo "  2. Login to HuggingFace: huggingface-cli login"
echo "  3. Request access to facebook/sam3 model"
echo "  4. Build the container: sbatch docker/build_apptainer.sbatch"
echo "  5. Run the webapp: sbatch docker/run_webapp_apptainer.sbatch"
echo ""
