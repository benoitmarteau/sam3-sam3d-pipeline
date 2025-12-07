"""
utils3d API Compatibility Layer

The newer utils3d (v1.5+) renamed/changed several functions.
This module provides compatibility wrappers and monkey-patches utils3d
to work with code written for the older API.

Changes handled:
- numpy:
    depth_edge -> depth_map_edge
    normals_edge -> normal_map_edge
    points_to_normals -> point_map_to_normal_map
    image_uv -> uv_map
    image_mesh -> build_mesh_from_map

- torch:
    perspective_from_fov_xy -> perspective_from_fov (with keyword args)
    intrinsics_from_fov_xy -> intrinsics_from_fov (with keyword args)
    rasterize_triangle_faces -> rasterize_triangles (different signature)
    compute_edges -> mesh_edges (returns only edges, not tuple)
    compute_connected_components -> mesh_connected_components
    compute_dual_graph -> removed (reimplemented here)
    compute_edge_connected_components -> removed (reimplemented here)
    remove_unreferenced_vertices -> remove_unused_vertices
"""

import numpy as np
import torch
import utils3d
import utils3d.numpy
import utils3d.torch


# =============================================================================
# NumPy compatibility wrappers
# =============================================================================

def depth_edge(depth, rtol=None, mask=None, **kwargs):
    """Compatibility wrapper for depth_map_edge."""
    return utils3d.numpy.depth_map_edge(depth, rtol=rtol, mask=mask, **kwargs)


def normals_edge(normals, tol=None, mask=None, **kwargs):
    """Compatibility wrapper for normal_map_edge."""
    return utils3d.numpy.normal_map_edge(normals, tol=tol, mask=mask, **kwargs)


def points_to_normals(pointmap, mask=None, **kwargs):
    """Compatibility wrapper for point_map_to_normal_map.

    Returns (normals, mask) tuple for compatibility with original API.
    """
    normals = utils3d.numpy.point_map_to_normal_map(pointmap, mask=mask, **kwargs)
    # Create a mask for valid normals (non-NaN)
    normals_mask = ~np.isnan(normals).any(axis=-1)
    if mask is not None:
        normals_mask = normals_mask & mask
    return normals, normals_mask


def image_uv(width, height, **kwargs):
    """Compatibility wrapper for uv_map."""
    return utils3d.numpy.uv_map(height, width, **kwargs)


def image_mesh(*args, mask=None, tri=False, **kwargs):
    """Compatibility wrapper for build_mesh_from_map."""
    return utils3d.numpy.build_mesh_from_map(*args, mask=mask, tri=tri, **kwargs)


# =============================================================================
# Torch compatibility wrappers
# =============================================================================

def _perspective_from_fov_xy(fov_x, fov_y, near, far):
    """Compatibility wrapper for perspective_from_fov_xy -> perspective_from_fov."""
    return utils3d.torch.perspective_from_fov(fov_x=fov_x, fov_y=fov_y, near=near, far=far)


def _intrinsics_from_fov_xy(fov_x, fov_y):
    """Compatibility wrapper for intrinsics_from_fov_xy -> intrinsics_from_fov.

    The old API took positional args (fov_x, fov_y).
    The new API uses keyword args: intrinsics_from_fov(fov_x=, fov_y=, ...).
    """
    return utils3d.torch.intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)


def _rasterize_triangle_faces(ctx, vertices, faces, width, height, uv=None, view=None, projection=None):
    """Compatibility wrapper for rasterize_triangle_faces -> rasterize_triangles.

    Original API returned dict with 'face_id', 'mask', 'uv', 'uv_dr' keys.
    New API returns tuple: (face_id, barycentric, derivatives_optional)
    """
    # New API expects vertices with batch dimension (B, V, 3)
    if vertices.dim() == 2:
        verts = vertices.unsqueeze(0)
    else:
        verts = vertices

    # Prepare attributes (UV coordinates if provided)
    if uv is not None:
        if uv.dim() == 2:
            attrs = uv.unsqueeze(0)
        else:
            attrs = uv
    else:
        attrs = None

    # Call new API
    face_id, bary, derivatives = utils3d.torch.rasterize_triangles(
        (height, width),
        vertices=verts,
        faces=faces,
        attributes=attrs,
        view=view,
        projection=projection,
        return_image_derivatives=True,
        ctx=ctx,
    )

    # Build result dict compatible with old API
    mask = (face_id >= 0).float()

    result = {
        'face_id': face_id.unsqueeze(0),
        'mask': mask.unsqueeze(0),
    }

    if attrs is not None:
        result['uv'] = bary.unsqueeze(0)
        result['uv_dr'] = derivatives.unsqueeze(0) if derivatives is not None else None

    return result


def _compute_edges(faces):
    """Compatibility wrapper for compute_edges -> mesh_edges."""
    edges = utils3d.torch.mesh_edges(faces)

    # Compute face2edge mapping
    num_faces = faces.shape[0]
    face2edge = torch.zeros((num_faces, 3), dtype=torch.long, device=faces.device)

    # Create edge lookup
    edge_set = {}
    for i, (v0, v1) in enumerate(edges.tolist()):
        key = (min(v0, v1), max(v0, v1))
        edge_set[key] = i

    for fi, face in enumerate(faces.tolist()):
        for ei, (v0, v1) in enumerate([(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]):
            key = (min(v0, v1), max(v0, v1))
            face2edge[fi, ei] = edge_set.get(key, -1)

    # Compute edge degrees
    edge_degrees = torch.zeros(edges.shape[0], dtype=torch.int32, device=faces.device)
    for fi in range(num_faces):
        for ei in range(3):
            edge_idx = face2edge[fi, ei].item()
            if edge_idx >= 0:
                edge_degrees[edge_idx] += 1

    return edges, face2edge, edge_degrees


def _compute_connected_components(faces, edges=None, face2edge=None):
    """Compute connected components of mesh faces."""
    return utils3d.torch.mesh_connected_components(faces)


def _compute_dual_graph(face2edge):
    """Compute dual graph of the mesh (faces as nodes, shared edges as edges)."""
    num_faces = face2edge.shape[0]
    edge_to_faces = {}

    for fi in range(num_faces):
        for ei in range(3):
            edge_idx = face2edge[fi, ei].item()
            if edge_idx >= 0:
                if edge_idx not in edge_to_faces:
                    edge_to_faces[edge_idx] = []
                edge_to_faces[edge_idx].append(fi)

    dual_edges = []
    dual_edge2edge = []

    for edge_idx, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            dual_edges.append(face_list)
            dual_edge2edge.append(edge_idx)

    if dual_edges:
        dual_edges = torch.tensor(dual_edges, dtype=torch.long, device=face2edge.device)
        dual_edge2edge = torch.tensor(dual_edge2edge, dtype=torch.long, device=face2edge.device)
    else:
        dual_edges = torch.zeros((0, 2), dtype=torch.long, device=face2edge.device)
        dual_edge2edge = torch.zeros((0,), dtype=torch.long, device=face2edge.device)

    return dual_edges, dual_edge2edge


def _compute_edge_connected_components(edges):
    """Compute connected components from edge list."""
    if edges.shape[0] == 0:
        return []

    vertices = torch.unique(edges.flatten())
    vertex_map = {v.item(): i for i, v in enumerate(vertices)}

    parent = list(range(len(vertices)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for e in edges:
        v0, v1 = vertex_map[e[0].item()], vertex_map[e[1].item()]
        union(v0, v1)

    components = {}
    for ei, e in enumerate(edges):
        root = find(vertex_map[e[0].item()])
        if root not in components:
            components[root] = []
        components[root].append(ei)

    return [torch.tensor(indices, dtype=torch.long, device=edges.device) for indices in components.values()]


def _remove_unreferenced_vertices(faces, vertices):
    """Remove vertices not referenced by any face."""
    return utils3d.torch.remove_unused_vertices(vertices, faces)


# =============================================================================
# Apply monkey patches
# =============================================================================

def apply_patches():
    """Apply all compatibility patches to utils3d."""
    # NumPy patches
    utils3d.numpy.depth_edge = depth_edge
    utils3d.numpy.normals_edge = normals_edge
    utils3d.numpy.points_to_normals = points_to_normals
    utils3d.numpy.image_uv = image_uv
    utils3d.numpy.image_mesh = image_mesh

    # Also add np.acos for older numpy compatibility
    if not hasattr(np, 'acos'):
        np.acos = np.arccos

    # Torch patches
    utils3d.torch.perspective_from_fov_xy = _perspective_from_fov_xy
    utils3d.torch.intrinsics_from_fov_xy = _intrinsics_from_fov_xy
    utils3d.torch.rasterize_triangle_faces = _rasterize_triangle_faces
    utils3d.torch.compute_edges = _compute_edges
    utils3d.torch.compute_connected_components = _compute_connected_components
    utils3d.torch.compute_dual_graph = _compute_dual_graph
    utils3d.torch.compute_edge_connected_components = _compute_edge_connected_components
    utils3d.torch.remove_unreferenced_vertices = _remove_unreferenced_vertices

    print("[utils3d_compat] Applied compatibility patches for utils3d API")


# Auto-apply patches when imported
apply_patches()
