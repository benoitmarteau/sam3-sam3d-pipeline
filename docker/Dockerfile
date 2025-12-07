# =============================================================================
# Dockerfile for SAM3 + SAM3D Pipeline
# =============================================================================
# This builds a Docker image from the NVIDIA PyTorch 24.10 base.
#
# Build:
#   docker build -t sam3d:latest -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8000:8000 -v $(pwd):/app sam3d:latest
#
# BASE IMAGE: nvcr.io/nvidia/pytorch:24.10-py3
#   - Python 3.10
#   - PyTorch 2.9.1 with CUDA 12.8 (supports torch.nn.attention APIs)
#   - torchvision, torchaudio
#   - NumPy, SciPy, and many ML libraries
#
# WHY PyTorch 2.5+:
#   SAM3 requires torch.nn.attention.sdpa_kernel (introduced in PyTorch 2.5).
#   24.10 has PyTorch 2.9.1 which fully supports these APIs.
#
# NOTE: Pre-built kaolin and flash_attn have ABI issues with 2.9.1.
#       Set SPARSE_ATTN_BACKEND=sdpa to use PyTorch native attention instead.
#
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.10-py3

LABEL maintainer="SAM3D Pipeline"
LABEL version="1.0.0"
LABEL description="SAM3 + SAM3D 3D Object Extraction Pipeline (PyTorch 2.5+)"

# =============================================================================
# Environment Variables
# =============================================================================
# CUDA Configuration
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# PyTorch CUDA Architecture (H200 = 9.0, A100 = 8.0)
# Note: Volta (7.0) support dropped in 24.10
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# Application settings
ENV GRADIO_ANALYTICS_ENABLED=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Skip SAM3D init module (not present in repo)
ENV LIDRA_SKIP_INIT=1

# nvdiffrast JIT compilation directory
ENV NVDIFFRAST_CACHE_DIR=/tmp/nvdiffrast_cache

# Suppress warnings
ENV HYDRA_FULL_ERROR=1

# Required by Meta's inference.py (expects conda environment)
ENV CONDA_PREFIX=/usr/local/cuda

# Default port
ENV PORT=8000

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxmu6 \
    xvfb \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Upgrade pip and essential build tools
# =============================================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Record PyTorch version from base container
# =============================================================================
RUN echo ">>> Recording base PyTorch version..." && \
    python -c "import torch; print(f'Base PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python -c "from torch.nn.attention import sdpa_kernel, SDPBackend; print('torch.nn.attention: OK')"

# =============================================================================
# Pin NumPy to 1.x (NumPy 2.0 breaks binary compatibility with many packages)
# =============================================================================
RUN pip install --no-cache-dir "numpy>=1.24,<2.0"

# =============================================================================
# Core Python Dependencies
# =============================================================================
RUN pip install --no-cache-dir \
    pillow>=10.0.0 \
    scipy \
    scikit-image \
    matplotlib \
    tqdm \
    einops \
    omegaconf \
    hydra-core==1.3.2 \
    loguru==0.7.2 \
    trimesh \
    imageio \
    imageio-ffmpeg \
    astor

# =============================================================================
# PyTorch3D Installation
# =============================================================================
RUN pip install --no-cache-dir fvcore iopath && \
    echo ">>> Building PyTorch3D from source (this takes ~10-15 minutes)..." && \
    pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    || echo ">>> WARNING: PyTorch3D installation failed"

# =============================================================================
# nvdiffrast Installation (for texture baking)
# =============================================================================
RUN echo ">>> Installing nvdiffrast..." && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/NVlabs/nvdiffrast.git \
    || echo ">>> WARNING: nvdiffrast installation failed"

# =============================================================================
# SAM3D Core Dependencies (Phase 1 - pip packages that may change PyTorch)
# =============================================================================
RUN echo ">>> Installing SAM3D dependencies (Phase 1)..." && \
    pip install --no-cache-dir \
        "timm>=1.0.17" \
        "transformers>=4.36.0,<4.50" \
        "accelerate>=0.25.0,<1.0" \
        safetensors \
        "lightning==2.3.3"

# Sparse convolution for 3D processing
RUN echo ">>> Installing spconv..." && \
    pip install --no-cache-dir spconv-cu120==2.3.8 \
    || pip install --no-cache-dir spconv-cu121==2.3.8 \
    || echo ">>> WARNING: spconv installation failed"

# MoGe for monocular geometry estimation (Microsoft)
RUN echo ">>> Installing MoGe..." && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/microsoft/MoGe.git \
    || echo ">>> WARNING: MoGe installation failed"

# Additional 3D dependencies (from SAM3D requirements.txt)
RUN echo ">>> Installing additional 3D dependencies..." && \
    pip install --no-cache-dir \
        roma \
        point-cloud-utils \
        open3d==0.18.0 \
        seaborn \
        xatlas==0.0.9 \
        pyvista \
        pymeshfix \
        python-igraph \
        easydict \
        plyfile

# utils3d for 3D utilities (used by SAM3D for texture baking)
RUN echo ">>> Installing utils3d..." && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/EasternJournalist/utils3d.git \
    || echo ">>> WARNING: utils3d installation failed"

# =============================================================================
# SAM3 Dependencies (from pyproject.toml)
# =============================================================================
RUN pip install --no-cache-dir \
    "timm>=1.0.17" \
    "numpy>=1.24,<2.0" \
    ftfy==6.1.1 \
    regex \
    "iopath>=0.1.10" \
    typing_extensions \
    huggingface_hub \
    decord

# =============================================================================
# Web Application Dependencies
# =============================================================================
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiofiles

# Gradio for alternative UI (optional)
RUN pip install --no-cache-dir gradio>=4.0.0 || echo ">>> gradio install failed"

# =============================================================================
# Re-pin NumPy (Gradio may have upgraded it to 2.x)
# =============================================================================
RUN pip install --no-cache-dir "numpy>=1.24,<2.0" --force-reinstall

# =============================================================================
# Fix OpenCV installation conflicts
# =============================================================================
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ 2>/dev/null || true && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.0"

# =============================================================================
# Record FINAL PyTorch version after all pip installs
# =============================================================================
RUN echo ">>> Recording final PyTorch version for CUDA extension builds..." && \
    python -c "import torch; print(f'Final PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python -c "from torch.nn.attention import sdpa_kernel, SDPBackend; print('torch.nn.attention: OK')"

# =============================================================================
# CUDA Extension Builds (Phase 2 - must be LAST!)
# =============================================================================
RUN echo ">>> Installing xformers..." && \
    pip install --no-cache-dir xformers --no-deps \
    || echo ">>> WARNING: xformers install failed"

RUN echo ">>> Installing gsplat from source..." && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/nerfstudio-project/gsplat.git@v1.0.0 \
    || pip install --no-cache-dir gsplat==1.0.0 \
    || echo ">>> WARNING: gsplat installation failed"

# Flash attention - NOTE: Takes 20-30 minutes to build
RUN echo ">>> Installing flash-attn from source (this takes ~20-30 minutes)..." && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2 \
    || echo ">>> WARNING: flash-attn installation failed - will use SDPA fallback"

# Kaolin - NOTE: Takes 15-20 minutes to build
RUN echo ">>> Installing Kaolin from source (this takes ~15-20 minutes)..." && \
    pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0 \
    || echo ">>> WARNING: Kaolin installation failed"

# =============================================================================
# Create app directory structure
# =============================================================================
RUN mkdir -p /app/webapp_outputs /app/logs

WORKDIR /app

# =============================================================================
# Verify installations
# =============================================================================
RUN echo ">>> Verifying key installations..." && \
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || echo ">>> NumPy: FAILED" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo ">>> PyTorch: FAILED" && \
    python -c "from torch.nn.attention import sdpa_kernel; print('torch.nn.attention: OK')" || echo ">>> torch.nn.attention: FAILED" && \
    python -c "import pytorch3d; print('PyTorch3D: OK')" || echo ">>> PyTorch3D: not installed" && \
    python -c "import spconv; print('spconv: OK')" || echo ">>> spconv: not installed" && \
    python -c "import kaolin; print('kaolin: OK')" || echo ">>> kaolin: not installed" && \
    python -c "import flash_attn; print('flash_attn: OK')" || echo ">>> flash_attn: not installed" && \
    python -c "import xformers; print('xformers: OK')" || echo ">>> xformers: not installed" && \
    python -c "import transformers; print(f'transformers: {transformers.__version__}')" || echo ">>> transformers: FAILED" && \
    python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || echo ">>> FastAPI: FAILED" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" || echo ">>> OpenCV: FAILED" && \
    echo ">>> Build complete!"

# =============================================================================
# Expose port and set entrypoint
# =============================================================================
EXPOSE 8000

# Default command runs the webapp
CMD ["python", "-u", "/app/webapp/app.py"]
