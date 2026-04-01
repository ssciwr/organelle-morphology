import logging
from dataclasses import dataclass, field
from typing import List, Dict, Union
import numpy as np
from dask import delayed
from dask.base import compute
import trimesh
from trimesh.intersections import mesh_plane

logger = logging.getLogger(__name__)


@dataclass
class ProfileData:
    """Stores the calculated 2D profile perimeters for a single organelle."""

    organelle_id: str
    axis_used: Union[str, tuple]
    num_slices_attempted: int
    perimeters: List[float] = field(default_factory=list)

    @property
    def mean_perimeter(self) -> float:
        return float(np.mean(self.perimeters)) if self.perimeters else np.nan


@delayed
def _slice_and_perimeter(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray):
    """Slices a mesh with a plane and returns the total perimeter."""
    try:
        lines = mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    except Exception as e:
        logger.debug(f"Mesh slicing failed: {e}")
        return np.nan

    if len(lines) == 0:
        return np.nan

    return np.sum(np.linalg.norm(lines[:, 1, :] - lines[:, 0, :], axis=1))


class ProfileCalculator:
    """Calculates 2D profile metrics for organelles within a project."""

    def __init__(self, project):
        self.project = project

    def _get_bounds(self, org):
        """Helper to extract bounding box limits."""
        geo = org.geometric_data
        if "voxel_bbox" not in geo:
            self.project.logger.warning(f"Missing voxel_bbox for {org.id}. Skipping.")
            return None, None

        # bbox is (min_row, min_col, min_depth, max_row, max_col, max_depth)
        res = org.source.resolution
        return np.array(geo["voxel_bbox"][:3]) * res, np.array(
            geo["voxel_bbox"][3:]
        ) * res

    def _parse_axis(self, axis):
        """Helper to parse string aliases (default is z) or vectors."""
        if isinstance(axis, str):
            return np.array(
                {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0]}.get(
                    axis.lower(), [0.0, 0.0, 1.0]
                )
            )
        vec = np.array(axis)
        return vec / np.linalg.norm(vec)

    def _compute_and_format(
        self, all_tasks, axis_record, num_slices
    ) -> Dict[str, ProfileData]:
        """Helper to compute Dask tasks and package them into ProfileData."""
        results = {}
        for org_id, tasks in all_tasks.items():
            perimeters = [p for p in compute(*tasks) if not np.isnan(p)]
            results[org_id] = ProfileData(org_id, axis_record, num_slices, perimeters)
        return results

    def calculate_profile_lengths(
        self, ids="er_*", axis="z", num_slices=20
    ) -> Dict[str, ProfileData]:
        """Calculates 2D profile perimeters along a given axis."""
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

            # Project bounds onto normal and sort to find limits
            min_proj, max_proj = sorted(
                [np.dot(min_b, plane_normal), np.dot(max_b, plane_normal)]
            )
            steps = np.linspace(min_proj, max_proj, num_slices + 2)[1:-1]

            all_tasks[org.id] = [
                _slice_and_perimeter(org.mesh, step * plane_normal, plane_normal)
                for step in steps
            ]

        axis_record = axis if isinstance(axis, str) else tuple(axis)
        return self._compute_and_format(all_tasks, axis_record, num_slices)

    def calculate_random_profiles(
        self, ids="er_*", num_planes=20, seed=None
    ) -> Dict[str, ProfileData]:
        """Calculates 2D profile perimeters using randomly oriented planes."""
        organelles = self.project.get_organelles(ids=ids)
        rng = np.random.default_rng(seed=seed)
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
                # Generate random normalized vector
                vec = rng.standard_normal(3)
                plane_normal = vec / np.linalg.norm(vec)

                min_proj, max_proj = sorted(
                    [np.dot(min_b, plane_normal), np.dot(max_b, plane_normal)]
                )

                # Random origin, minus 5% of edges to avoid misses
                margin = (max_proj - min_proj) * 0.05
                origin = (
                    rng.uniform(min_proj + margin, max_proj - margin) * plane_normal
                )

                org_tasks.append(_slice_and_perimeter(org.mesh, origin, plane_normal))

            all_tasks[org.id] = org_tasks

        return self._compute_and_format(all_tasks, "random", num_planes)
