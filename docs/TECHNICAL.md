# Technical Documentation

## Architecture Overview

```
User Input (Image + Text Prompt)
    ↓
FastAPI Web Server (webapp/app.py) on port 8000
    ↓
SAM3SAM3DPipeline (pipeline.py)
    ├─ Stage 1: SAM3 Segmentation (HuggingFace model)
    │      └─ Text-based instance segmentation
    │      └─ Returns masks for ALL matching objects
    ├─ Stage 2: SAM3D 3D Reconstruction (local checkpoints)
    │      └─ Per-object Gaussian splat generation
    │      └─ Mesh extraction via marching cubes
    └─ Stage 3: Scene Composition (multi-object merge)
           └─ Gaussian splat merging (PLY)
           └─ Mesh merging with transforms (GLB)
    ↓
Output Files (.ply, .glb, .obj, .gif)
```

## Container Environment

### Base Image
- **Image**: `nvcr.io/nvidia/pytorch:24.10-py3`
- **PyTorch**: 2.5.0 (required for `torch.nn.attention.sdpa_kernel`)
- **CUDA**: 12.6
- **Python**: 3.10

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.0 | Deep learning framework (NGC pinned) |
| PyTorch3D | 0.7.8 | 3D deep learning utilities |
| kaolin | 0.17.0 | NVIDIA 3D deep learning (built from source) |
| flash_attn | 2.7.2 | Efficient attention (built from source) |
| spconv | 2.3.8 | Sparse convolution for 3D |
| transformers | 4.36+ | HuggingFace model loading |
| trimesh | latest | Mesh manipulation |
| gsplat | 1.0.0 | Gaussian splatting rendering |

### Why Build from Source?

CUDA extensions like `kaolin` and `flash_attn` must be built from source to match the exact PyTorch version and CUDA toolkit. Pre-built wheels often have ABI (Application Binary Interface) mismatches that cause crashes.

The container build order is critical:
1. Install pip packages that may change PyTorch version (gradio, etc.)
2. Record final PyTorch version
3. Build CUDA extensions (kaolin, flash_attn) LAST
4. Never pip install anything after CUDA extension builds

## API Compatibility Layer

### The Problem

SAM3D was written for an older version of `utils3d`. The newer utils3d (v1.5+) renamed several functions, causing import errors.

### The Solution

`lib/utils3d_compat.py` monkey-patches utils3d to provide backward compatibility:

**NumPy API Mapping:**
| Old Name | New Name |
|----------|----------|
| `depth_edge` | `depth_map_edge` |
| `normals_edge` | `normal_map_edge` |
| `points_to_normals` | `point_map_to_normal_map` |
| `image_uv` | `uv_map` |
| `image_mesh` | `build_mesh_from_map` |

**Torch API Mapping:**
| Old Name | New Name/Changes |
|----------|-----------------|
| `perspective_from_fov_xy(fov_x, fov_y, near, far)` | `perspective_from_fov(fov_x=, fov_y=, near=, far=)` |
| `intrinsics_from_fov_xy(fov_x, fov_y)` | `intrinsics_from_fov(fov_x=, fov_y=)` |
| `rasterize_triangle_faces(ctx, verts, ...)` → dict | `rasterize_triangles(size, vertices=, ...)` → tuple |
| `compute_edges(faces)` → (edges, face2edge, degrees) | `mesh_edges(faces)` → edges only |
| `compute_connected_components(faces)` | `mesh_connected_components(faces)` |
| `remove_unreferenced_vertices(faces, verts)` | `remove_unused_vertices(verts, faces)` (arg order!) |

The patches are auto-applied when `lib.utils3d_compat` is imported.

## Multi-Object Scene Composition

### Gaussian Splats (PLY)

SAM3D's `make_scene()` function handles Gaussian splat composition:
1. Each object's Gaussians are in local space
2. `compose_transform(scale, rotation, translation)` creates a Transform3d
3. `transform_points()` moves Gaussians to world space
4. Gaussians are concatenated into a single splat cloud

### Mesh Scene (GLB)

For GLB export, `create_scene_mesh()` applies the same transforms:
1. Load individual GLB meshes (already in Y-up space)
2. Convert mesh vertices to torch tensors
3. Apply SAM3D's `compose_transform` + `transform_points`
4. Concatenate meshes into a trimesh.Scene
5. Export as GLB

**Important**: Object positions in multi-object scenes are approximate. SAM3D reconstructs each object independently and provides layout estimates, but these don't perfectly preserve original spatial relationships. See [GitHub Issue #53](https://github.com/facebookresearch/sam-3d-objects/issues/53).

## Environment Variables

### Container Runtime

| Variable | Value | Purpose |
|----------|-------|---------|
| `SPARSE_ATTN_BACKEND` | `sdpa` | Use PyTorch native attention (fallback) |
| `ATTN_BACKEND` | `sdpa` | Same as above |
| `SSL_CERT_FILE` | `""` | Prevent gradio SSL errors |
| `CONDA_PREFIX` | `/usr/local/cuda` | Required by SAM3D inference.py |
| `HYDRA_FULL_ERROR` | `1` | Full error traces |
| `TORCH_HOME` | `/root/.cache/torch` | Mounted cache for DINO models |
| `TORCH_EXTENSIONS_DIR` | `/root/.cache/torch_extensions` | Mounted cache for JIT extensions |
| `HF_HOME` | `/root/.cache/huggingface` | Mounted cache for HuggingFace |

### Build Time

| Variable | Value | Purpose |
|----------|-------|---------|
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;9.0` | Target GPU architectures |
| `FORCE_CUDA` | `1` | Force CUDA compilation |

## SLURM Integration

### Container Build Job

```bash
#SBATCH -J sam3d_build
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
```

The build job:
1. Sets up Apptainer cache directories
2. Builds the container from `sam3d.def`
3. Logs progress to `logs/`

### Webapp Run Job

```bash
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
```

The run job:
1. Checks for container image
2. Sets up bind mounts for code and caches
3. Launches container with GPU passthrough
4. Prints SSH tunnel instructions

## Known Issues & Workarounds

### Issue 1: nvdiffrast JIT Compilation

**Problem**: nvdiffrast needs to compile CUDA extensions at runtime, but the container can't find the host's GCC.

**Workaround**: Pipeline detects rasterization errors and falls back to:
- Vertex colors instead of UV-mapped textures
- Disabled mesh postprocessing

### Issue 2: Gradio Updates PyTorch

**Problem**: Installing Gradio can pull in a newer PyTorch, breaking CUDA extensions.

**Solution**: Container build installs Gradio BEFORE building CUDA extensions.

### Issue 3: NumPy 2.0 Incompatibility

**Problem**: NumPy 2.0 breaks binary compatibility with many packages.

**Solution**: Container pins NumPy to 1.x (`numpy>=1.24,<2.0`).

### Issue 4: SSL Certificate Errors

**Problem**: SLURM sets `SSL_CERT_FILE` to an invalid path.

**Solution**: Container clears the variable: `--env SSL_CERT_FILE=""`

## Performance Notes

- **GPU Memory**: 32GB+ recommended for typical images
- **Processing Time**: 30-90 seconds per object (H200)
- **Container Size**: ~11GB
- **Build Time**: 30-60 minutes

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Container Installation

```bash
apptainer exec --nv sam3d.sif python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import kaolin; print('kaolin: OK')
import flash_attn; print('flash_attn: OK')
"
```

### Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```
