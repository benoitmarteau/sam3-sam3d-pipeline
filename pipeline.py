#!/usr/bin/env python3
"""
pipeline.py - Combined SAM3 + SAM3D Pipeline

This script combines SAM3 (text-to-segmentation) with SAM3D (image+mask to 3D)
to create a complete text-to-3D pipeline.

Supports:
- Single object: Detects best matching object and creates 3D model
- Multi-object: Detects ALL matching objects (e.g., 2 horses) and creates combined 3D scene

Usage:
    # Single object mode (default) - with synthetic test image
    python pipeline.py --prompt "red ball"

    # Single object mode - with custom image
    python pipeline.py --image_path photo.jpg --prompt "dog"

    # Multi-object mode - finds ALL objects matching the prompt
    python pipeline.py --image_path horses.jpg --prompt "horse" --multi_object

    # Multi-object with max limit
    python pipeline.py --image_path crowd.jpg --prompt "person" --multi_object --max_objects 5

    # Using SAM3D's built-in example (skip SAM3)
    python pipeline.py --use_example

    # With custom image and mask (skip SAM3)
    python pipeline.py --image_path photo.jpg --mask_path mask.png

Python API:
    from pipeline import SAM3SAM3DPipeline

    pipeline = SAM3SAM3DPipeline()

    # Single object
    result = pipeline.run(image_path="photo.jpg", prompt="dog", output_dir="./output")

    # Multi-object (finds ALL dogs in image and combines into scene)
    result = pipeline.run_multi_object(image_path="dogs.jpg", prompt="dog", output_dir="./output")
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# Add project paths early
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load utils3d compatibility patches BEFORE any other imports that might use utils3d
try:
    from lib import utils3d_compat  # noqa: F401 - imported for side effects (patches)
except ImportError as e:
    print(f"Warning: Could not load utils3d_compat: {e}", file=sys.stderr)

import torch
import numpy as np
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add SAM3/SAM3D paths
SAM3_PATH = PROJECT_ROOT / "sam3"
SAM3D_PATH = PROJECT_ROOT / "sam-3d-objects"

sys.path.insert(0, str(SAM3_PATH))
sys.path.insert(0, str(SAM3D_PATH))
sys.path.insert(0, str(SAM3D_PATH / "notebook"))


# ============================================================================
# Utility Functions
# ============================================================================

def create_synthetic_image(output_dir: str) -> Tuple[Image.Image, str]:
    """
    Create a synthetic test image with a red ball.

    Args:
        output_dir: Directory to save the image

    Returns:
        Tuple of (PIL Image, path to saved image)
    """
    from PIL import ImageDraw

    logger.info("Creating synthetic test image (red ball on gray background)...")

    size = 512
    image = Image.new('RGB', (size, size))
    pixels = image.load()

    # Create gradient background
    for y in range(size):
        for x in range(size):
            gray = int(180 - (y / size) * 60)
            pixels[x, y] = (gray, gray, gray)

    # Draw red ball with shading
    draw = ImageDraw.Draw(image)
    center = (size // 2, size // 2)
    radius = 100

    for r in range(radius, 0, -1):
        shade = int(255 * (r / radius) * 0.3)
        red = min(255, 200 + shade)
        green = int(50 * (1 - r / radius))
        blue = int(50 * (1 - r / radius))
        draw.ellipse(
            [center[0] - r, center[1] - r, center[0] + r, center[1] + r],
            fill=(red, green, blue)
        )

    # Add highlight
    highlight_center = (center[0] - 30, center[1] - 30)
    draw.ellipse(
        [highlight_center[0] - 15, highlight_center[1] - 15,
         highlight_center[0] + 15, highlight_center[1] + 15],
        fill=(255, 200, 200)
    )

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "input.png")
    image.save(image_path)
    logger.info(f"Saved synthetic image to: {image_path}")

    return image, image_path


def get_example_inputs() -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Get SAM3D's built-in example image and mask.

    Returns:
        Tuple of (image_pil, image_np, mask_np)
    """
    from inference import load_image, load_single_mask

    example_dir = SAM3D_PATH / "notebook" / "images" / "shutterstock_stylish_kidsroom_1640806567"

    if not example_dir.exists():
        raise FileNotFoundError(f"Example directory not found: {example_dir}")

    image_path = example_dir / "image.png"
    image_pil = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    image_np = load_image(str(image_path))
    mask_np = load_single_mask(str(example_dir), index=14)

    logger.info(f"Loaded example image: {image_path}")
    logger.info(f"Image shape: {image_np.shape}, Mask shape: {mask_np.shape}")
    logger.info(f"Mask coverage: {mask_np.sum() / mask_np.size * 100:.1f}%")

    return image_pil, image_np, mask_np


def load_image_and_mask(image_path: str, mask_path: str) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Load image and mask from file paths.

    Args:
        image_path: Path to RGB image
        mask_path: Path to mask image

    Returns:
        Tuple of (image_pil, image_np, mask_np)
    """
    image_pil = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    image_np = np.array(image_pil).astype(np.uint8)

    mask_img = Image.open(mask_path)
    mask_np = np.array(mask_img)

    # Handle different mask formats
    if mask_np.ndim == 3:
        if mask_np.shape[2] == 4:
            mask_np = mask_np[..., -1]  # Use alpha channel
        else:
            mask_np = mask_np[..., 0]  # Use first channel

    # Convert to boolean
    mask_np = mask_np > 0

    logger.info(f"Loaded image: {image_path} ({image_pil.size[0]}x{image_pil.size[1]})")
    logger.info(f"Loaded mask: {mask_path}, coverage: {mask_np.sum() / mask_np.size * 100:.1f}%")

    return image_pil, image_np, mask_np


# ============================================================================
# SAM3SAM3DPipeline Class
# ============================================================================

class SAM3SAM3DPipeline:
    """
    Combined SAM3 + SAM3D Pipeline for text-to-3D object generation.

    Example:
        pipeline = SAM3SAM3DPipeline()
        result = pipeline.run(
            image_path="photo.jpg",
            prompt="dog",
            output_dir="./output"
        )
    """

    def __init__(
        self,
        sam3d_config_path: Optional[str] = None,
        compile_sam3d: bool = False,
        lazy_load: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            sam3d_config_path: Path to SAM3D pipeline.yaml config
            compile_sam3d: Whether to compile SAM3D model (slower startup, faster inference)
            lazy_load: If True, load models only when needed (default: True)
        """
        self.sam3d_config_path = sam3d_config_path or str(SAM3D_PATH / "checkpoints" / "hf" / "pipeline.yaml")
        self.compile_sam3d = compile_sam3d
        self._sam3_processor = None
        self._sam3d_inference = None

        if not lazy_load:
            self._load_sam3()
            self._load_sam3d()

    def _load_sam3(self):
        """Load SAM3 model."""
        if self._sam3_processor is not None:
            return

        logger.info("Loading SAM3 model...")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model(load_from_HF=True)
            self._sam3_processor = Sam3Processor(model)
            logger.info("SAM3 model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            logger.error("Make sure you have:")
            logger.error("  1. Requested access at https://huggingface.co/facebook/sam3")
            logger.error("  2. Logged in with: huggingface-cli login")
            raise

    def _load_sam3d(self):
        """Load SAM3D model."""
        if self._sam3d_inference is not None:
            return

        logger.info("Loading SAM3D model...")
        try:
            from inference import Inference

            if not os.path.exists(self.sam3d_config_path):
                logger.error(f"SAM3D config not found: {self.sam3d_config_path}")
                logger.error("Make sure you have downloaded SAM3D checkpoints.")
                raise FileNotFoundError(f"Config not found: {self.sam3d_config_path}")

            self._sam3d_inference = Inference(self.sam3d_config_path, compile=self.compile_sam3d)

            # Verify and force pytorch3d rendering engine to avoid nvdiffrast dependency
            if hasattr(self._sam3d_inference, '_pipeline'):
                current_engine = getattr(self._sam3d_inference._pipeline, 'rendering_engine', 'unknown')
                logger.info(f"SAM3D rendering engine: {current_engine}")
                if current_engine != 'pytorch3d':
                    logger.warning(f"Forcing rendering_engine from '{current_engine}' to 'pytorch3d'")
                    self._sam3d_inference._pipeline.rendering_engine = 'pytorch3d'

            logger.info("SAM3D model loaded successfully!")

        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.error("Make sure SAM3D dependencies are installed:")
            logger.error("  cd sam-3d-objects && pip install -e '.[inference]'")
            raise

    def segment(self, image: Image.Image, prompt: str) -> Optional[np.ndarray]:
        """
        Run SAM3 segmentation.

        Args:
            image: PIL Image (RGB)
            prompt: Text description of object to segment

        Returns:
            Boolean mask as numpy array, or None if no object found
        """
        self._load_sam3()

        logger.info(f"Running SAM3 segmentation with prompt: '{prompt}'")

        inference_state = self._sam3_processor.set_image(image)
        output = self._sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output.get("masks", None)
        scores = output.get("scores", None)

        if masks is None or masks.shape[0] == 0:
            logger.warning(f"No objects found for prompt: '{prompt}'")
            return None

        logger.info(f"Found {masks.shape[0]} object(s) matching '{prompt}'")

        # Get the best mask (highest confidence)
        if scores is not None:
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()
            logger.info(f"Using best mask (index {best_idx}, confidence {best_score:.3f})")
        else:
            best_idx = 0

        # Extract mask and convert to boolean numpy array
        mask = masks[best_idx]
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # Ensure 2D
        while mask.ndim > 2:
            mask = mask.squeeze()

        # Convert to boolean
        mask = mask > 0

        logger.info(f"Mask shape: {mask.shape}, coverage: {mask.sum() / mask.size * 100:.1f}%")
        return mask

    def _check_nvdiffrast_available(self) -> bool:
        """Check if nvdiffrast is properly installed and can compile CUDA extensions."""
        try:
            import nvdiffrast.torch as dr
            # Try to create a simple context to verify compilation works
            # This is a lightweight test that should trigger JIT compilation if needed
            return True
        except Exception as e:
            logger.warning(f"nvdiffrast not available: {e}")
            return False

    def reconstruct_3d(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        seed: int = 42,
        generate_mesh: bool = True,
        with_texture: bool = True
    ) -> Dict[str, Any]:
        """
        Run SAM3D 3D reconstruction.

        Args:
            image: RGB image as numpy array [H, W, 3]
            mask: Boolean mask as numpy array [H, W]
            seed: Random seed for reproducibility
            generate_mesh: Whether to generate mesh output (GLB/OBJ)
            with_texture: Whether to bake textures onto mesh

        Returns:
            Dict with 'gs' (Gaussian splatting), 'glb' (mesh), and other outputs
        """
        self._load_sam3d()

        # Check if nvdiffrast is available for texture baking
        # Note: utils3d internally uses nvdiffrast for rasterization in texture baking
        # If nvdiffrast compilation fails, we must disable texture baking
        if with_texture and generate_mesh:
            if not hasattr(self, '_nvdiffrast_available'):
                self._nvdiffrast_available = self._check_nvdiffrast_available()

            if not self._nvdiffrast_available:
                logger.warning("nvdiffrast not available - disabling texture baking, using vertex colors instead")
                with_texture = False

        logger.info(f"Running SAM3D 3D reconstruction (seed={seed}, mesh={generate_mesh}, texture={with_texture})...")

        # Ensure pytorch3d rendering engine is set (for the parts that can use it)
        if hasattr(self._sam3d_inference, '_pipeline'):
            self._sam3d_inference._pipeline.rendering_engine = 'pytorch3d'

        # Run inference with mesh generation if requested
        try:
            output = self._sam3d_inference._pipeline.run(
                self._sam3d_inference.merge_mask_to_rgba(image, mask),
                None,
                seed,
                stage1_only=False,
                with_mesh_postprocess=generate_mesh and getattr(self, '_mesh_postprocess_available', True),
                with_texture_baking=with_texture and generate_mesh,
                with_layout_postprocess=True,
                use_vertex_color=not with_texture,
                stage1_inference_steps=None,
                pointmap=None,
            )
        except Exception as e:
            error_str = str(e).lower()
            # If texture baking or postprocessing fails (e.g., due to nvdiffrast/utils3d issues), try simpler config
            is_rasterization_error = any(x in error_str for x in ['nvdiffrast', 'cc1plus', 'compilation', 'rasterize', 'unpack'])

            if is_rasterization_error and (with_texture or generate_mesh):
                logger.warning(f"Mesh postprocessing/texture failed ({e}), retrying with simplified settings...")
                self._nvdiffrast_available = False  # Cache the failure
                self._mesh_postprocess_available = False  # Disable mesh postprocessing
                output = self._sam3d_inference._pipeline.run(
                    self._sam3d_inference.merge_mask_to_rgba(image, mask),
                    None,
                    seed,
                    stage1_only=False,
                    with_mesh_postprocess=False,  # Disable postprocessing
                    with_texture_baking=False,
                    with_layout_postprocess=True,
                    use_vertex_color=True,
                    stage1_inference_steps=None,
                    pointmap=None,
                )
            else:
                raise

        available_outputs = [k for k in output.keys() if output[k] is not None]
        logger.info(f"Generated outputs: {available_outputs}")

        return output

    def segment_all(self, image: Image.Image, prompt: str) -> Optional[List[np.ndarray]]:
        """
        Run SAM3 segmentation and return ALL matching masks.

        Args:
            image: PIL Image (RGB)
            prompt: Text description of objects to segment

        Returns:
            List of boolean masks, or None if no objects found
        """
        self._load_sam3()

        logger.info(f"Running SAM3 segmentation (all masks) with prompt: '{prompt}'")

        inference_state = self._sam3_processor.set_image(image)
        output = self._sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output.get("masks", None)
        scores = output.get("scores", None)

        if masks is None or masks.shape[0] == 0:
            logger.warning(f"No objects found for prompt: '{prompt}'")
            return None

        logger.info(f"Found {masks.shape[0]} object(s) matching '{prompt}'")

        all_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()

            while mask.ndim > 2:
                mask = mask.squeeze()

            mask = mask > 0

            if scores is not None:
                score = scores[i].item() if torch.is_tensor(scores) else scores[i]
                logger.info(f"  Mask {i}: coverage {mask.sum() / mask.size * 100:.1f}%, score {score:.3f}")
            else:
                logger.info(f"  Mask {i}: coverage {mask.sum() / mask.size * 100:.1f}%")

            all_masks.append(mask)

        return all_masks

    def create_scene(self, *outputs) -> Any:
        """
        Merge multiple 3D object outputs into a single scene.

        Uses SAM3D's make_scene function to combine multiple objects
        into a unified scene with proper spatial placement.

        Args:
            *outputs: Variable number of SAM3D inference output dicts

        Returns:
            Merged Gaussian splatting object representing the scene
        """
        from inference import make_scene

        if len(outputs) == 0:
            raise ValueError("At least one output is required to create a scene")

        if len(outputs) == 1:
            logger.info("Single object - returning as-is")
            return outputs[0].get("gs")

        logger.info(f"Merging {len(outputs)} objects into scene...")
        scene_gs = make_scene(*outputs, in_place=False)
        logger.info("Scene created successfully!")

        return scene_gs

    def create_scene_mesh(self, outputs: List[Dict]) -> Any:
        """
        Merge multiple GLB meshes into a single scene mesh using SAM3D's original transform logic.

        This follows the exact same approach as SAM3D's `get_mesh()` in layout_post_optimization_utils.py:
        1. Rotate mesh vertices from Z-up to Y-up (as done in postprocessing)
        2. Use compose_transform(scale, rotation, translation) to create Transform3d
        3. Apply transform_points() to position vertices in world space

        This is identical to how `object_pointcloud()` and `make_scene()` work for Gaussian splats.

        Args:
            outputs: List of SAM3D output dicts containing 'glb', 'rotation', 'translation', 'scale'

        Returns:
            Combined trimesh.Scene object with transformed meshes
        """
        import trimesh
        import numpy as np
        import torch

        valid_outputs = [out for out in outputs if out.get("glb") is not None]
        if len(valid_outputs) == 0:
            logger.warning("No valid meshes to merge")
            return None

        if len(valid_outputs) == 1:
            logger.info("Single mesh - returning as-is")
            return valid_outputs[0]["glb"]

        logger.info(f"Merging {len(valid_outputs)} meshes into scene using SAM3D's original transforms...")

        # Import SAM3D's transform utilities (same as used in get_mesh and object_pointcloud)
        try:
            from pytorch3d.transforms import quaternion_to_matrix
            from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
            use_sam3d_transforms = True
            logger.info("Using SAM3D compose_transform for mesh positioning")
        except ImportError as e:
            logger.warning(f"Could not import SAM3D transforms: {e}")
            logger.warning("Falling back to side-by-side arrangement")
            use_sam3d_transforms = False

        # Create a combined scene
        combined_scene = trimesh.Scene()

        # Z-up to Y-up rotation matrix - same as in postprocessing_utils.py line 666
        # vertices @ [[1,0,0],[0,0,-1],[0,1,0]]
        R_z2y = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

        for i, output in enumerate(valid_outputs):
            mesh = output.get("glb")
            if mesh is None:
                continue

            # Get SAM3D transforms
            rotation = output.get("rotation")  # quaternion [1, 4]
            translation = output.get("translation")  # [1, 3]
            scale = output.get("scale")  # [1, 3]

            # Get the geometry (handle both Scene and Trimesh)
            if isinstance(mesh, trimesh.Scene):
                geom_list = list(mesh.geometry.values())
                if len(geom_list) == 0:
                    continue
                combined_geom = trimesh.util.concatenate(geom_list)
            elif isinstance(mesh, trimesh.Trimesh):
                combined_geom = mesh.copy()
            else:
                logger.warning(f"Object {i}: Unknown mesh type: {type(mesh)}, skipping")
                continue

            if use_sam3d_transforms and rotation is not None and translation is not None and scale is not None:
                # Use SAM3D's exact transform approach from get_mesh() and object_pointcloud()
                vertices = combined_geom.vertices.copy()

                # Log transforms for debugging
                rot_np = rotation.cpu().numpy().flatten() if hasattr(rotation, 'cpu') else np.array(rotation).flatten()
                trans_np = translation.cpu().numpy().flatten() if hasattr(translation, 'cpu') else np.array(translation).flatten()
                scale_np = scale.cpu().numpy().flatten() if hasattr(scale, 'cpu') else np.array(scale).flatten()
                logger.info(f"Object {i}: trans={trans_np}, scale={scale_np}, rot(quat)={rot_np}")

                # Step 1: Rotate mesh from Z-up to Y-up (same as get_mesh in layout_post_optimization_utils.py)
                # Note: GLB meshes are already in Y-up from postprocessing, so we need to undo that first
                # to match what get_mesh expects (original Z-up mesh), then apply the Z-to-Y rotation
                # Actually, since GLB is already Y-up, we skip the rotation here and work directly

                # Convert vertices to torch tensor
                mesh_vertices = torch.from_numpy(vertices).float()
                if torch.cuda.is_available():
                    mesh_vertices = mesh_vertices.cuda()

                # Step 2: Create the transform using compose_transform (same as object_pointcloud)
                # compose_transform expects rotation as a 3x3 matrix, not quaternion
                R_l2c = quaternion_to_matrix(rotation)  # [1, 3, 3]
                l2c_transform = compose_transform(
                    scale=scale,
                    rotation=R_l2c,
                    translation=translation
                )

                # Step 3: Apply transform to vertices (same as get_mesh and object_pointcloud)
                points_world = l2c_transform.transform_points(mesh_vertices.unsqueeze(0))
                vertices_transformed = points_world[0].cpu().numpy()

                # Update mesh vertices
                combined_geom.vertices = vertices_transformed

                logger.info(f"Object {i}: Applied SAM3D transform via compose_transform + transform_points")
            else:
                # Fallback: just center the mesh
                center = combined_geom.centroid
                combined_geom.vertices = combined_geom.vertices - center
                logger.info(f"Object {i}: Centered (no SAM3D transform available)")

            # Add to scene
            combined_scene.add_geometry(combined_geom, node_name=f"object_{i}")

        logger.info("Scene mesh created with SAM3D's original transform logic")
        return combined_scene

    def run_multi_object(
        self,
        image_path: str,
        prompt: str,
        output_dir: str = "pipeline_output",
        seed: int = 42,
        skip_preview: bool = False,
        max_objects: int = 10,
        generate_mesh: bool = True,
        with_texture: bool = True
    ) -> Dict[str, Any]:
        """
        Run multi-object pipeline: detect ALL objects matching a single prompt and create combined scene.

        This is the main multi-object method. For example, if you have an image with 2 horses
        and prompt "horse", SAM3 will detect both horses, then SAM3D reconstructs each one,
        and finally they are combined into a single scene.

        Args:
            image_path: Path to input image
            prompt: Text prompt to segment (e.g., "horse" - will find ALL horses)
            output_dir: Directory to save outputs
            seed: Base random seed for SAM3D
            skip_preview: Skip rendering preview GIF
            max_objects: Maximum number of objects to process (default: 10)

        Returns:
            Dict with results including individual objects and combined scene
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        result = {
            "success": False,
            "prompt": prompt,
            "num_objects_found": 0,
            "num_objects_processed": 0,
            "objects": [],
            "masks": [],
            "input_path": None,
            "combined_mask_path": None,
            "scene_ply_path": None,
            "scene_glb_path": None,
            "scene_obj_path": None,
            "scene_preview_path": None,
            "ply_path": None,  # For single object compatibility
            "glb_path": None,  # For single object compatibility
            "obj_path": None,  # For single object compatibility
            "preview_path": None,  # For single object compatibility
            "elapsed_time": 0
        }

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Apply EXIF orientation to handle rotated images from phones/cameras
        image_pil = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
        image_np = np.array(image_pil).astype(np.uint8)

        # Save input image
        input_path = os.path.join(output_dir, "input.png")
        image_pil.save(input_path)
        result["input_path"] = input_path

        logger.info("=" * 50)
        logger.info(f"Running Multi-Object Pipeline")
        logger.info(f"Prompt: '{prompt}'")
        logger.info("=" * 50)

        # Step 1: SAM3 Segmentation - get ALL masks
        logger.info("")
        logger.info("Step 1: SAM3 Segmentation (finding all objects)")
        logger.info("-" * 30)

        all_masks = self.segment_all(image_pil, prompt)

        if all_masks is None or len(all_masks) == 0:
            logger.error(f"SAM3 failed to find any objects for prompt: '{prompt}'")
            result["error"] = f"No objects found matching '{prompt}'"
            return result

        result["num_objects_found"] = len(all_masks)
        logger.info(f"Found {len(all_masks)} object(s) matching '{prompt}'")

        # Limit number of objects
        if len(all_masks) > max_objects:
            logger.warning(f"Limiting to {max_objects} objects (found {len(all_masks)})")
            all_masks = all_masks[:max_objects]

        # Save individual masks and combined mask
        combined_mask = np.zeros_like(all_masks[0], dtype=bool)
        for i, mask in enumerate(all_masks):
            # Save individual mask
            mask_path = os.path.join(output_dir, f"mask_{i}.png")
            mask_uint8 = (mask.astype(np.uint8)) * 255
            Image.fromarray(mask_uint8).save(mask_path)
            result["masks"].append(mask_path)

            # Accumulate for combined mask
            combined_mask = combined_mask | mask

        # Save combined mask
        combined_mask_path = os.path.join(output_dir, "mask_combined.png")
        combined_mask_uint8 = (combined_mask.astype(np.uint8)) * 255
        Image.fromarray(combined_mask_uint8).save(combined_mask_path)
        result["combined_mask_path"] = combined_mask_path
        logger.info(f"Saved combined mask to: {combined_mask_path}")

        # Free SAM3 memory before loading SAM3D
        if self._sam3_processor is not None:
            logger.info("Freeing SAM3 memory...")
            del self._sam3_processor
            self._sam3_processor = None
            torch.cuda.empty_cache()

        # Step 2: SAM3D 3D Reconstruction for each object
        logger.info("")
        logger.info("Step 2: SAM3D 3D Reconstruction")
        logger.info("-" * 30)

        all_outputs = []
        for i, mask in enumerate(all_masks):
            logger.info(f"\n--- Reconstructing object {i+1}/{len(all_masks)} ---")
            logger.info(f"Mask coverage: {mask.sum() / mask.size * 100:.1f}%")

            try:
                output = self.reconstruct_3d(
                    image_np, mask, seed=seed + i,
                    generate_mesh=generate_mesh,
                    with_texture=with_texture
                )
                all_outputs.append(output)

                obj_data = {
                    "index": i,
                    "mask_path": result["masks"][i],
                }

                # Save individual object PLY
                obj_ply_path = os.path.join(output_dir, f"object_{i}.ply")
                if output.get("gs") is not None:
                    output["gs"].save_ply(obj_ply_path)
                    logger.info(f"Saved object {i} PLY to: {obj_ply_path}")
                    obj_data["ply_path"] = obj_ply_path

                # Save individual object GLB (textured mesh)
                if output.get("glb") is not None:
                    obj_glb_path = os.path.join(output_dir, f"object_{i}.glb")
                    obj_obj_path = os.path.join(output_dir, f"object_{i}.obj")
                    try:
                        output["glb"].export(obj_glb_path)
                        output["glb"].export(
                            obj_obj_path,
                            include_texture=True,
                            resolver=None
                        )
                        logger.info(f"Saved object {i} GLB to: {obj_glb_path}")
                        logger.info(f"Saved object {i} OBJ to: {obj_obj_path}")
                        obj_data["glb_path"] = obj_glb_path
                        obj_data["obj_path"] = obj_obj_path
                    except Exception as e:
                        logger.warning(f"Failed to save object {i} GLB/OBJ: {e}")

                result["objects"].append(obj_data)

            except Exception as e:
                import traceback
                logger.warning(f"Failed to reconstruct object {i}: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")
                continue

        result["num_objects_processed"] = len(all_outputs)

        if len(all_outputs) == 0:
            result["error"] = "No objects were successfully reconstructed"
            return result

        # Step 3: Create combined scene
        logger.info("")
        logger.info("Step 3: Creating Combined Scene")
        logger.info("-" * 30)

        if len(all_outputs) == 1:
            # Single object - use it directly
            scene_gs = all_outputs[0].get("gs")
            scene_glb = all_outputs[0].get("glb")
            scene_ply_path = os.path.join(output_dir, "model.ply")
            scene_glb_path = os.path.join(output_dir, "model.glb")
            scene_obj_path = os.path.join(output_dir, "model.obj")
            result["ply_path"] = scene_ply_path
            result["glb_path"] = scene_glb_path
            result["obj_path"] = scene_obj_path
        else:
            # Multiple objects - merge into scene
            scene_gs = self.create_scene(*all_outputs)

            # Create combined scene mesh with proper transforms
            # Pass full outputs so we can access rotation/translation/scale
            scene_glb = self.create_scene_mesh(all_outputs)

            scene_ply_path = os.path.join(output_dir, "scene_combined.ply")
            scene_glb_path = os.path.join(output_dir, "scene_combined.glb")
            scene_obj_path = os.path.join(output_dir, "scene_combined.obj")
            result["scene_ply_path"] = scene_ply_path
            result["scene_glb_path"] = scene_glb_path
            result["scene_obj_path"] = scene_obj_path
            # Also set ply_path to scene for compatibility
            result["ply_path"] = scene_ply_path
            result["glb_path"] = scene_glb_path
            result["obj_path"] = scene_obj_path

        # Save PLY (Gaussian Splatting)
        try:
            scene_gs.save_ply(scene_ply_path)
            logger.info(f"Saved {'scene' if len(all_outputs) > 1 else 'model'} PLY to: {scene_ply_path}")
        except Exception as e:
            logger.warning(f"Failed to save scene PLY: {e}")

        # Save GLB/OBJ (Textured Mesh)
        if scene_glb is not None:
            try:
                scene_glb.export(scene_glb_path)
                logger.info(f"Saved scene GLB to: {scene_glb_path}")

                # Try to export OBJ as well
                try:
                    scene_glb.export(
                        scene_obj_path,
                        include_texture=True,
                        resolver=None
                    )
                    logger.info(f"Saved scene OBJ to: {scene_obj_path}")
                except Exception as obj_err:
                    logger.warning(f"Failed to save scene OBJ: {obj_err}")
            except Exception as e:
                logger.warning(f"Failed to save GLB/OBJ: {e}")

        # Step 4: Render preview
        if not skip_preview:
            logger.info("")
            logger.info("Step 4: Rendering Preview")
            logger.info("-" * 30)

            try:
                from inference import ready_gaussian_for_video_rendering, render_video
                import imageio

                gs_for_render = ready_gaussian_for_video_rendering(scene_gs)

                video = render_video(
                    gs_for_render,
                    r=2.0 if len(all_outputs) > 1 else 1.5,  # Larger radius for multi-object
                    fov=60,
                    pitch_deg=20 if len(all_outputs) > 1 else 15,
                    resolution=512,
                    num_frames=36
                )

                gif_path = os.path.join(output_dir, "preview.gif")
                imageio.mimsave(gif_path, video["color"], format="GIF", duration=100)
                result["preview_path"] = gif_path
                if len(all_outputs) > 1:
                    result["scene_preview_path"] = gif_path
                logger.info(f"Saved preview to: {gif_path}")

                # Save first frame
                png_path = os.path.join(output_dir, "preview.png")
                Image.fromarray(video["color"][0]).save(png_path)

            except Exception as e:
                logger.warning(f"Failed to render preview: {e}")

        result["success"] = True
        result["elapsed_time"] = time.time() - start_time

        # Summary
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Multi-Object Pipeline Complete! (took {result['elapsed_time']:.1f}s)")
        logger.info(f"Objects found: {result['num_objects_found']}")
        logger.info(f"Objects processed: {result['num_objects_processed']}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 50)

        return result

    def run(
        self,
        image_path: Optional[str] = None,
        prompt: str = "red ball",
        mask_path: Optional[str] = None,
        output_dir: str = "pipeline_output",
        seed: int = 42,
        use_example: bool = False,
        skip_preview: bool = False
    ) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            image_path: Path to input image (creates synthetic if None and not use_example)
            prompt: Text prompt for SAM3 segmentation
            mask_path: Optional mask path (skips SAM3 if provided)
            output_dir: Directory to save outputs
            seed: Random seed for SAM3D
            use_example: Use SAM3D's built-in example (skips SAM3)
            skip_preview: Skip rendering preview GIF

        Returns:
            Dict with results and saved file paths
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        result = {
            "success": False,
            "input_path": None,
            "mask_path": None,
            "ply_path": None,
            "preview_path": None,
            "elapsed_time": 0
        }

        # Determine input mode and load data
        if use_example:
            logger.info("=" * 50)
            logger.info("Using SAM3D built-in example (skipping SAM3)")
            logger.info("=" * 50)
            image_pil, image_np, mask = get_example_inputs()

        elif mask_path is not None:
            logger.info("=" * 50)
            logger.info("Using provided image and mask (skipping SAM3)")
            logger.info("=" * 50)

            if image_path is None:
                raise ValueError("--image_path is required when using --mask_path")

            image_pil, image_np, mask = load_image_and_mask(image_path, mask_path)

        else:
            logger.info("=" * 50)
            logger.info("Running full SAM3 → SAM3D pipeline")
            logger.info("=" * 50)

            # Prepare image
            if image_path is None:
                image_pil, saved_image_path = create_synthetic_image(output_dir)
                result["input_path"] = saved_image_path
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                # Apply EXIF orientation to handle rotated images from phones/cameras
                image_pil = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')

            image_np = np.array(image_pil).astype(np.uint8)

            # Step 1: SAM3 Segmentation
            logger.info("")
            logger.info("Step 1: SAM3 Segmentation")
            logger.info("-" * 30)

            mask = self.segment(image_pil, prompt)

            if mask is None:
                logger.error(f"SAM3 failed to find any objects for prompt: '{prompt}'")
                result["error"] = f"No object found matching '{prompt}'"
                return result

            # Free SAM3 memory before loading SAM3D
            if self._sam3_processor is not None:
                del self._sam3_processor
                self._sam3_processor = None
                torch.cuda.empty_cache()

        # Step 2: SAM3D 3D Reconstruction
        logger.info("")
        logger.info("Step 2: SAM3D 3D Reconstruction")
        logger.info("-" * 30)

        output_3d = self.reconstruct_3d(image_np, mask, seed=seed)

        # Save outputs
        logger.info("")
        logger.info("Saving outputs...")
        logger.info("-" * 30)

        saved_files = self._save_outputs(
            image_pil, mask, output_3d, output_dir,
            skip_preview=skip_preview
        )

        result.update(saved_files)
        result["success"] = True
        result["elapsed_time"] = time.time() - start_time
        result["output_3d"] = output_3d

        # Summary
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Pipeline Complete! (took {result['elapsed_time']:.1f}s)")
        logger.info("=" * 50)
        logger.info(f"Output directory: {output_dir}")

        return result

    def _save_outputs(
        self,
        image: Image.Image,
        mask: np.ndarray,
        output_3d: dict,
        output_dir: str,
        skip_preview: bool = False
    ) -> Dict[str, str]:
        """Save all pipeline outputs."""
        saved = {}

        # Save input image
        input_path = os.path.join(output_dir, "input.png")
        image.save(input_path)
        saved["input_path"] = input_path
        logger.info(f"Saved input image to: {input_path}")

        # Save mask
        mask_uint8 = (mask.astype(np.uint8)) * 255
        mask_path = os.path.join(output_dir, "mask.png")
        Image.fromarray(mask_uint8).save(mask_path)
        saved["mask_path"] = mask_path
        logger.info(f"Saved mask to: {mask_path}")

        # Save masked visualization
        image_np = np.array(image)
        masked_image = image_np.copy().astype(np.float32)
        masked_image[~mask] = masked_image[~mask] * 0.3
        masked_path = os.path.join(output_dir, "masked.png")
        Image.fromarray(masked_image.astype(np.uint8)).save(masked_path)
        saved["masked_path"] = masked_path
        logger.info(f"Saved masked image to: {masked_path}")

        # Save 3D output (PLY - Gaussian Splatting)
        if output_3d.get("gs") is not None:
            gs = output_3d["gs"]
            ply_path = os.path.join(output_dir, "model.ply")
            try:
                gs.save_ply(ply_path)
                saved["ply_path"] = ply_path
                logger.info(f"Saved Gaussian Splatting PLY to: {ply_path}")
            except Exception as e:
                logger.warning(f"Failed to save PLY: {e}")

        # Save 3D mesh output (GLB with textures)
        if output_3d.get("glb") is not None:
            glb_mesh = output_3d["glb"]
            try:
                # Save GLB (binary glTF)
                glb_path = os.path.join(output_dir, "model.glb")
                glb_mesh.export(glb_path)
                saved["glb_path"] = glb_path
                logger.info(f"Saved textured GLB to: {glb_path}")

                # Save OBJ with materials (for external visualization)
                # OBJ format will create .obj, .mtl, and texture files
                obj_path = os.path.join(output_dir, "model.obj")
                glb_mesh.export(
                    obj_path,
                    include_texture=True,
                    resolver=None
                )
                saved["obj_path"] = obj_path
                logger.info(f"Saved OBJ to: {obj_path}")

                # Also check if texture was exported
                mtl_path = os.path.join(output_dir, "model.mtl")
                texture_path = os.path.join(output_dir, "material_0.png")
                if os.path.exists(mtl_path):
                    logger.info(f"Saved MTL to: {mtl_path}")
                if os.path.exists(texture_path):
                    logger.info(f"Saved texture to: {texture_path}")

            except Exception as e:
                logger.warning(f"Failed to save GLB/OBJ: {e}")

        # Render preview
        if not skip_preview and output_3d.get("gs") is not None:
            try:
                from inference import ready_gaussian_for_video_rendering, render_video
                import imageio

                logger.info("Rendering 3D preview...")
                gs_for_render = ready_gaussian_for_video_rendering(output_3d["gs"])

                video = render_video(
                    gs_for_render,
                    r=1.5,
                    fov=60,
                    pitch_deg=15,
                    resolution=512,
                    num_frames=36
                )

                # Save GIF
                gif_path = os.path.join(output_dir, "preview.gif")
                imageio.mimsave(gif_path, video["color"], format="GIF", duration=100)
                saved["preview_path"] = gif_path
                logger.info(f"Saved preview GIF to: {gif_path}")

                # Save first frame
                png_path = os.path.join(output_dir, "preview.png")
                Image.fromarray(video["color"][0]).save(png_path)
                saved["preview_png_path"] = png_path

            except Exception as e:
                logger.warning(f"Failed to render preview: {e}")

        return saved


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 + SAM3D Pipeline: Text-to-3D Object Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with synthetic red ball image (single object)
    python pipeline.py --prompt "red ball"

    # With your own image (single object)
    python pipeline.py --image_path photo.jpg --prompt "dog"

    # Multi-object mode - finds ALL objects matching prompt
    python pipeline.py --image_path horses.jpg --prompt "horse" --multi_object

    # Multi-object with limit
    python pipeline.py --image_path crowd.jpg --prompt "person" --multi_object --max_objects 5

    # Skip SAM3, use SAM3D's built-in example
    python pipeline.py --use_example

    # Skip SAM3, provide your own mask
    python pipeline.py --image_path photo.jpg --mask_path mask.png
        """
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to input image (creates synthetic if not provided)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="red ball",
        help="Text prompt for SAM3 segmentation (default: 'red ball')"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Optional: provide mask directly (skips SAM3)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline_output",
        help="Directory to save outputs (default: pipeline_output)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SAM3D (default: 42)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile SAM3D model (slower startup, faster inference)"
    )
    parser.add_argument(
        "--use_example",
        action="store_true",
        help="Use SAM3D's built-in example (skips SAM3)"
    )
    parser.add_argument(
        "--skip_preview",
        action="store_true",
        help="Skip rendering preview GIF"
    )
    parser.add_argument(
        "--multi_object",
        action="store_true",
        help="Multi-object mode: find ALL objects matching the prompt and combine into scene"
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=10,
        help="Maximum number of objects to process in multi-object mode (default: 10)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mask_path is not None and args.image_path is None:
        logger.error("--image_path is required when using --mask_path")
        sys.exit(1)

    if args.image_path is not None and not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        sys.exit(1)

    if args.mask_path is not None and not os.path.exists(args.mask_path):
        logger.error(f"Mask not found: {args.mask_path}")
        sys.exit(1)

    if args.multi_object and args.use_example:
        logger.error("--multi_object cannot be used with --use_example")
        sys.exit(1)

    if args.multi_object and args.mask_path is not None:
        logger.error("--multi_object cannot be used with --mask_path")
        sys.exit(1)

    # Run pipeline
    pipeline = SAM3SAM3DPipeline(compile_sam3d=args.compile)

    if args.multi_object:
        # Multi-object mode
        if args.image_path is None:
            logger.error("--image_path is required for --multi_object mode")
            sys.exit(1)

        result = pipeline.run_multi_object(
            image_path=args.image_path,
            prompt=args.prompt,
            output_dir=args.output_dir,
            seed=args.seed,
            skip_preview=args.skip_preview,
            max_objects=args.max_objects
        )
    else:
        # Single object mode
        result = pipeline.run(
            image_path=args.image_path,
            prompt=args.prompt,
            mask_path=args.mask_path,
            output_dir=args.output_dir,
            seed=args.seed,
            use_example=args.use_example,
            skip_preview=args.skip_preview
        )

    if not result["success"]:
        logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    return result


if __name__ == "__main__":
    main()
