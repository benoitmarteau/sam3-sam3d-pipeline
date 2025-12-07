#!/usr/bin/env python3
"""
FastAPI Web Application for SAM3 → SAM3D Pipeline

This provides a REST API for the combined segmentation and 3D reconstruction pipeline.

Endpoints:
    POST /api/generate_3d - Upload image + text prompt → get 3D model
    GET /objects/<filename> - Serve generated 3D files
    GET / - Serve frontend UI

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
    # or
    python app.py
"""

import os
import sys

# Load utils3d compatibility patches BEFORE any other imports that might use utils3d
# This must be done early to ensure patches are applied before sam3d_objects is imported
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
try:
    from lib import utils3d_compat  # noqa: F401 - imported for side effects (patches)
except ImportError as e:
    print(f"Warning: Could not load utils3d_compat: {e}", file=sys.stderr)
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("SAM3-SAM3D-WebApp")

# ============================================================================
# Configuration
# ============================================================================

# Base directory (parent of webapp)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Output directory for generated files
OUTPUT_DIR = BASE_DIR / "webapp_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Static files directory (for frontend)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# SAM3 and SAM3D paths
SAM3_PATH = BASE_DIR / "sam3"
SAM3D_PATH = BASE_DIR / "sam-3d-objects"
SAM3D_CONFIG = SAM3D_PATH / "checkpoints" / "hf" / "pipeline.yaml"

# Add paths to Python path
sys.path.insert(0, str(SAM3_PATH))
sys.path.insert(0, str(SAM3D_PATH))
sys.path.insert(0, str(SAM3D_PATH / "notebook"))
sys.path.insert(0, str(BASE_DIR))

# Maximum file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="SAM3 → SAM3D 3D Object Generator",
    description="Generate 3D objects from images using text prompts",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Pipeline Instance (lazy loaded)
# ============================================================================

_pipeline = None


def get_pipeline():
    """Get or create pipeline instance (lazy loading)."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing SAM3 → SAM3D pipeline...")
        from pipeline import SAM3SAM3DPipeline
        _pipeline = SAM3SAM3DPipeline(
            sam3d_config_path=str(SAM3D_CONFIG) if SAM3D_CONFIG.exists() else None,
            compile_sam3d=False,
            lazy_load=True
        )
        logger.info("Pipeline initialized!")
    return _pipeline


# ============================================================================
# Response Models
# ============================================================================

class GenerationResponse(BaseModel):
    """Response model for 3D generation."""
    success: bool
    job_id: str
    message: str
    num_objects: int = 1
    glb_url: Optional[str] = None
    obj_url: Optional[str] = None
    ply_url: Optional[str] = None
    mask_url: Optional[str] = None
    preview_url: Optional[str] = None
    objects: List[dict] = []  # Individual object details for multi-object
    elapsed_time: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    pipeline_loaded: bool
    timestamp: str




# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        pipeline_loaded=_pipeline is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/generate_3d", response_model=GenerationResponse)
async def generate_3d(
    image: UploadFile = File(..., description="Input image file"),
    text_prompt: str = Form(..., description="Text description of object to segment"),
    seed: int = Form(42, description="Random seed for reproducibility"),
    max_objects: int = Form(50, description="Maximum number of objects to detect (max: 100)")
):
    """
    Generate 3D object(s) from an uploaded image and text prompt.

    This endpoint automatically detects ALL objects matching the prompt.
    For example, if you have an image with 2 horses and prompt "horse",
    it will detect both horses and create a combined 3D scene.

    Process:
    1. SAM3 segments ALL objects described by the text prompt
    2. SAM3D reconstructs the 3D geometry for each segmented object
    3. Multiple objects are combined into a single scene
    4. Returns URLs to download the generated files

    Args:
        image: Uploaded image file (JPG, PNG, etc.)
        text_prompt: Description of the object(s) to extract (e.g., "horse", "person")
        seed: Random seed for reproducible results
        max_objects: Maximum number of objects to process (default: 50, max: 100)

    Returns:
        JSON with URLs to generated 3D files and object details
    """
    job_id = str(uuid.uuid4())[:8]

    # Validate and cap max_objects
    max_objects = min(max(1, max_objects), 100)

    logger.info(f"[{job_id}] New generation request: prompt='{text_prompt}', max_objects={max_objects}")

    # Validate file extension
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Create job output directory
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # Save uploaded image
        image_path = job_dir / f"input{ext}"
        content = await image.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
            )

        with open(image_path, "wb") as f:
            f.write(content)

        logger.info(f"[{job_id}] Saved input image: {image_path}")

        # Get pipeline and run multi-object mode
        # This automatically handles both single and multiple objects
        pipeline = get_pipeline()

        result = pipeline.run_multi_object(
            image_path=str(image_path),
            prompt=text_prompt,
            output_dir=str(job_dir),
            seed=seed,
            skip_preview=False,
            max_objects=max_objects
        )

        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Generation failed")
            )

        # Build response with URLs
        base_url = f"/objects/{job_id}"
        num_objects = result.get("num_objects_processed", 1)

        # Determine message based on object count
        if num_objects == 1:
            message = f"Successfully generated 3D model for '{text_prompt}'"
        else:
            message = f"Successfully generated {num_objects} 3D objects for '{text_prompt}' (combined into scene)"

        response = GenerationResponse(
            success=True,
            job_id=job_id,
            message=message,
            num_objects=num_objects,
            elapsed_time=result.get("elapsed_time")
        )

        # Add main PLY URL (either single model or combined scene)
        if result.get("ply_path") and os.path.exists(result["ply_path"]):
            ply_filename = Path(result["ply_path"]).name
            response.ply_url = f"{base_url}/{ply_filename}"

        # Add GLB URL (textured mesh)
        if result.get("glb_path") and os.path.exists(result["glb_path"]):
            glb_filename = Path(result["glb_path"]).name
            response.glb_url = f"{base_url}/{glb_filename}"
        elif result.get("objects") and len(result["objects"]) > 0:
            # Fallback: use first object's GLB if scene GLB doesn't exist
            first_obj = result["objects"][0]
            if first_obj.get("glb_path") and os.path.exists(first_obj["glb_path"]):
                glb_filename = Path(first_obj["glb_path"]).name
                response.glb_url = f"{base_url}/{glb_filename}"

        # Add OBJ URL (textured mesh for external viewers)
        if result.get("obj_path") and os.path.exists(result["obj_path"]):
            obj_filename = Path(result["obj_path"]).name
            response.obj_url = f"{base_url}/{obj_filename}"

        # Add combined mask URL
        if result.get("combined_mask_path") and os.path.exists(result["combined_mask_path"]):
            response.mask_url = f"{base_url}/mask_combined.png"
        elif result.get("masks") and len(result["masks"]) > 0:
            # Single object - use first mask
            response.mask_url = f"{base_url}/mask_0.png"

        # Add preview URL
        if result.get("preview_path") and os.path.exists(result["preview_path"]):
            response.preview_url = f"{base_url}/preview.gif"

        # Add individual object details
        for obj in result.get("objects", []):
            obj_detail = {
                "index": obj.get("index", 0),
                "ply_url": f"{base_url}/{Path(obj['ply_path']).name}" if obj.get("ply_path") else None,
                "glb_url": f"{base_url}/{Path(obj['glb_path']).name}" if obj.get("glb_path") else None,
                "obj_url": f"{base_url}/{Path(obj['obj_path']).name}" if obj.get("obj_path") else None,
                "mask_url": f"{base_url}/{Path(obj['mask_path']).name}" if obj.get("mask_path") else None
            }
            response.objects.append(obj_detail)

        logger.info(f"[{job_id}] Generation complete! Objects: {num_objects}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{job_id}] Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/objects/{job_id}/{filename}")
async def serve_object(job_id: str, filename: str):
    """
    Serve generated 3D files.

    Args:
        job_id: The job ID from the generation request
        filename: Name of the file to serve (e.g., model.glb, model.ply)

    Returns:
        The requested file
    """
    # Sanitize inputs
    job_id = job_id.replace("/", "").replace("\\", "").replace("..", "")
    filename = filename.replace("/", "").replace("\\", "").replace("..", "")

    file_path = OUTPUT_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine content type
    ext = Path(filename).suffix.lower()
    content_types = {
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".ply": "application/octet-stream",
        ".obj": "model/obj",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=content_type,
        filename=filename
    )


@app.delete("/objects/{job_id}")
async def delete_job(job_id: str):
    """Delete generated files for a job."""
    job_id = job_id.replace("/", "").replace("\\", "").replace("..", "")
    job_dir = OUTPUT_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        shutil.rmtree(job_dir)
        return {"success": True, "message": f"Deleted job {job_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs")
async def list_jobs():
    """List all generated jobs."""
    jobs = []
    for job_dir in OUTPUT_DIR.iterdir():
        if job_dir.is_dir():
            files = [f.name for f in job_dir.iterdir() if f.is_file()]
            jobs.append({
                "job_id": job_dir.name,
                "files": files,
                "created": datetime.fromtimestamp(job_dir.stat().st_ctime).isoformat()
            })

    return {"jobs": sorted(jobs, key=lambda x: x["created"], reverse=True)}


# Note: The /api/generate_multi_object endpoint has been deprecated.
# Use /api/generate_3d instead - it automatically detects ALL objects
# matching the prompt (e.g., "horse" will find all horses in the image).


# ============================================================================
# Static Files
# ============================================================================

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable (for Docker/Slurm flexibility)
    port = int(os.environ.get("PORT", 8000))

    # Pre-load pipeline if desired (comment out for lazy loading)
    # get_pipeline()

    logger.info(f"Starting webapp on port {port}...")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1  # Single worker due to GPU memory
    )
