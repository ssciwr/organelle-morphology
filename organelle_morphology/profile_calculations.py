import logging
from dataclasses import dataclass
from typing import List, Union
import numpy as np
from dask import delayed
from dask.base import compute
import trimesh
from trimesh.intersections import mesh_plane
from organelle_morphology.statistics import Properties, Stats

logger = logging.getLogger(__name__)


@dataclass
class ProfileProperties(Properties):
    """Physical measurements of the 2D profile."""

    perimeters: List[float]
    widths: List[float]
    mean_perimeter: float
    mean_width: float


@dataclass
class ProfileMeta(Properties):
    """Context for the profile calculation."""

    organelle_id: str
    axis_used: Union[str, tuple]
    num_slices_attempted: int


@delayed
def _slice_profile(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray):
    """
    Slices a mesh with a plane.
    Returns: (perimeter, width) of the resulting 2D cross-section.
    """
    try:
        lines = mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    except Exception as e:
        logger.debug(f"Mesh slicing failed: {e}")
        return np.nan, np.nan

    if len(lines) == 0:
        return np.nan, np.nan

    # 1. Perimeter Calculation
    segment_vectors = lines[:, 1, :] - lines[:, 0, :]
    perimeter = np.sum(np.linalg.norm(segment_vectors, axis=1))

    # 2. Width Calculation (Longest diameter between any two points in the slice)
    pts = lines.reshape(-1, 3)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    width = np.max(np.linalg.norm(diffs, axis=-1))

    return perimeter, width


class ProfileCalculator:
    """Calculates 2D profile metrics for organelles within a project."""

    def __init__(self, project):
        self.project = project

    def _get_bounds(self, org):
        """Helper to safely extract real-world bounding box limits."""
        geo = org.geometric_data
        if "voxel_bbox" not in geo:
            self.project.logger.warning(f"Missing voxel_bbox for {org.id}. Skipping.")
            return None, None

        res = org.source.resolution
        return np.array(geo["voxel_bbox"][:3]) * res, np.array(
            geo["voxel_bbox"][3:]
        ) * res

    def _parse_axis(self, axis):
        """Helper to cleanly parse string aliases or raw vectors."""
        if isinstance(axis, str):
            return np.array(
                {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0]}.get(
                    axis.lower(), [0.0, 0.0, 1.0]
                )
            )
        vec = np.array(axis)
        return vec / np.linalg.norm(vec)

    def _compute_and_format(self, all_tasks, axis_record, num_slices=None) -> None:
        """
        Helper to compute Dask tasks and register them into the central
        Project.stats list.
        """
        for org_id, tasks in all_tasks.items():
            computed = compute(*tasks)

            perimeters = [res[0] for res in computed if not np.isnan(res[0])]
            widths = [res[1] for res in computed if not np.isnan(res[1])]
            attempted = num_slices if num_slices is not None else len(tasks)

            # Pre-calculate means for the dataclass fields
            m_perimeter = float(np.mean(perimeters)) if perimeters else np.nan
            m_width = float(np.mean(widths)) if widths else np.nan

            data = ProfileProperties(
                perimeters=perimeters,
                widths=widths,
                mean_perimeter=m_perimeter,
                mean_width=m_width,
            )
            meta = ProfileMeta(
                organelle_id=org_id,
                axis_used=axis_record,
                num_slices_attempted=attempted,
            )

            # Register the result in the central project repository
            self.project.add_stat(Stats(data=data, meta=meta))

    def calculate_profile_lengths(self, ids="er_*", axis="z", num_slices=20) -> None:
        """Calculates 2D profile metrics along a given fixed axis."""
        organelles = self.project.get_organelles(ids=ids)
        plane_normal = self._parse_axis(axis)
        all_tasks = {}

        self.project.logger.info(
            f"Generating {num_slices} {axis}-axis slices for {len(organelles)} organelles..."
        )

        for org in organelles:
            min_b, max_b = self._get_bounds(org)
            if min_b is None:
                continue

            min_proj, max_proj = sorted(
                [np.dot(min_b, plane_normal), np.dot(max_b, plane_normal)]
            )
            steps = np.linspace(min_proj, max_proj, num_slices + 2)[1:-1]

            all_tasks[org.id] = [
                _slice_profile(org.mesh, step * plane_normal, plane_normal)
                for step in steps
            ]

        axis_record = axis if isinstance(axis, str) else tuple(axis)
        self._compute_and_format(all_tasks, axis_record, num_slices)

    def calculate_random_profiles(self, ids="er_*", num_planes=20, seed=None) -> None:
        """Calculates 2D profile metrics using randomly oriented planes."""
        organelles = self.project.get_organelles(ids=ids)
        rng = np.random.default_rng(seed)
        all_tasks = {}

        self.project.logger.info(
            f"Generating {num_planes} random slices for {len(organelles)} organelles..."
        )

        for org in organelles:
            min_b, max_b = self._get_bounds(org)
            if min_b is None:
                continue

            org_tasks = []
            for _ in range(num_planes):
                vec = rng.standard_normal(3)
                plane_normal = vec / np.linalg.norm(vec)
                min_proj, max_proj = sorted(
                    [np.dot(min_b, plane_normal), np.dot(max_b, plane_normal)]
                )

                margin = (max_proj - min_proj) * 0.05
                origin = (
                    rng.uniform(min_proj + margin, max_proj - margin) * plane_normal
                )

                org_tasks.append(_slice_profile(org.mesh, origin, plane_normal))

            all_tasks[org.id] = org_tasks

        self._compute_and_format(all_tasks, "random", num_planes)

    def calculate_skeleton_profiles(self, ids="er_*", sample_distance=0.1) -> None:
        """
        Calculates 2D profile perimeters and widths perpendicular to the organelle skeleton.
        """
        organelles = self.project.get_organelles(ids=ids)
        all_tasks = {}

        self.project.logger.info(
            f"Generating skeleton-perpendicular slices for {len(organelles)} organelles..."
        )

        for org in organelles:
            if org.skeleton is None:
                self.project.logger.warning(
                    f"No skeleton found for {org.id}. Run skeletonize first. Skipping."
                )
                continue

            org_tasks = []

            for edge in org.skeleton.edges:
                p1 = np.array(org.skeleton.vertices[edge[0]])
                p2 = np.array(org.skeleton.vertices[edge[1]])

                edge_vec = p2 - p1
                length = np.linalg.norm(edge_vec)

                if length == 0:
                    continue

                plane_normal = edge_vec / length

                n_points = max(1, int(np.ceil(length / sample_distance)))
                factors = np.linspace(0, 1, n_points + 2)[1:-1]

                for f in factors:
                    origin = p1 + f * edge_vec
                    org_tasks.append(_slice_profile(org.mesh, origin, plane_normal))

            all_tasks[org.id] = org_tasks

        self._compute_and_format(all_tasks, "skeleton", num_slices=None)
