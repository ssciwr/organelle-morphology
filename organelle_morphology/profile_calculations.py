import logging
import numpy as np
from dask import delayed
from dask.base import compute
import trimesh
from trimesh.intersections import mesh_plane

logger = logging.getLogger(__name__)


@delayed
def _slice_and_perimeter(
    mesh: trimesh.Trimesh, plane_origin: np.ndarray, plane_normal: np.ndarray
):
    """
    Slices a mesh with a plane and returns the sum of perimeters of the resulting 2D cross-sections.
    """
    # mesh_plane returns intersecting line segments in 3D: (n, 2, 3)
    try:
        lines = mesh_plane(mesh, plane_normal=plane_normal, plane_origin=plane_origin)
    except Exception as e:
        logger.debug(f"Mesh slicing failed: {e}")
        return np.nan

    if len(lines) == 0:
        return np.nan

    # Calculate lengths of all line segments
    # lines shape is (n, 2, 3), representing n segments defined by 2 points in 3D space
    segment_vectors = lines[:, 1, :] - lines[:, 0, :]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    # Total perimeter is the sum of all segment lengths in this slice
    total_perimeter = np.sum(segment_lengths)
    return total_perimeter


def calculate_profile_lengths(project, ids="er_*", axis="z", num_slices=20):
    """
    Calculates 2D profile perimeters for specified organelles along a given axis.

    Args:
        project: The initialized Project instance.
        ids: Glob pattern for the organelles to analyze.
        axis: String ('x', 'y', 'z') or a 3D numpy array representing the plane normal.
        num_slices: Number of slices to take across the bounding box of the organelle.

    Returns:
        A dictionary mapping organelle IDs to a list of perimeter lengths.
    """
    organelles = project.get_organelles(ids=ids)

    # Define the normal vector based on the chosen axis
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    if isinstance(axis, str):
        plane_normal = axis_map.get(axis.lower(), np.array([0.0, 0.0, 1.0]))
    else:
        plane_normal = np.array(axis)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

    all_tasks = {}

    project.logger.info(
        f"Generating {num_slices} {axis}-axis slicing planes for {len(organelles)} organelles..."
    )

    for org in organelles:
        # Fetch pre-calculated basic geometric properties to get the bounding box bounds
        # This avoids computing the full mesh just to find where to put the planes
        geo_data = org.geometric_data

        if "voxel_bbox" not in geo_data:
            project.logger.warning(f"Missing voxel_bbox for {org.id}. Skipping.")
            continue

        # skimage bbox is (min_row, min_col, min_depth, max_row, max_col, max_depth)
        bbox = geo_data["voxel_bbox"]
        min_bound = np.array([bbox[0], bbox[1], bbox[2]]) * org.source.resolution
        max_bound = np.array([bbox[3], bbox[4], bbox[5]]) * org.source.resolution

        # Project bounds onto the normal vector to find extents along the axis
        min_proj = np.dot(min_bound, plane_normal)
        max_proj = np.dot(max_bound, plane_normal)

        if min_proj > max_proj:
            min_proj, max_proj = max_proj, min_proj

        # Generate origins (skip the absolute edges to ensure good cuts)
        steps = np.linspace(min_proj, max_proj, num_slices + 2)[1:-1]

        org_tasks = []
        for step in steps:
            # Origin is the step distance along the normal vector
            origin = step * plane_normal

            # Pass the delayed mesh to our delayed slicing function
            task = _slice_and_perimeter(org.mesh, origin, plane_normal)
            org_tasks.append(task)

        all_tasks[org.id] = org_tasks

    # Compute all perimeters in parallel
    results = {}
    for org_id, tasks in all_tasks.items():
        perimeters = compute(*tasks)
        # Filter out NaNs (slices that missed the mesh completely)
        results[org_id] = [p for p in perimeters if not np.isnan(p)]

    return results
