from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
import trimesh
from dask import delayed
from dask.base import compute
from scipy.spatial.distance import pdist
from trimesh.intersections import mesh_plane

from organelle_morphology.analysis import Analysis
from organelle_morphology.statistics import Properties, Stats

if TYPE_CHECKING:
    from organelle_morphology.project import Project

logger = logging.getLogger(__name__)


@dataclass
class ProfileProperties(Properties):
    """Physical measurements of the 2D profile."""

    perimeters: List[float]
    widths: List[float]
    ratios: List[float]  # width / perimeter ratio for every slice
    mean_perimeter: float  # mean across all slices for one organelle
    mean_width: float
    mean_ratio: float


@dataclass
class ProfileMeta(Properties):
    """Context for the profile calculation."""

    organelle_id: str
    axis_used: Union[str, tuple]
    num_slices_attempted: int


@delayed
def _slice_profile(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray):
    """
    Slices a mesh and returns a list of (perimeter, width) (for each
    disjoint component found in the slice).
    """
    try:
        lines = mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    except Exception as e:
        logger.debug(f"Mesh slicing failed: {e}")
        return []

    if len(lines) == 0:
        return []

    # One organelle might be cut twice by one plane
    # -> use trimesh to group segments into individual loops
    path = trimesh.load_path(lines)
    components = []
    for discrete_path in path.discrete:
        # perimeter of this specific path
        diffs = np.diff(discrete_path, axis=0)
        perimeter = np.sum(np.linalg.norm(diffs, axis=1))
        if perimeter <= 0:
            logger.debug(f"Perimeter calculation failed for path: {discrete_path}")
            continue

        # width of this specific path (Max distance)
        if len(discrete_path) <= 1:
            logger.debug(f"Width calculation failed for path: {discrete_path}")
            continue

        # pdist calculates the pairwise distances between points in discrete_path
        width = pdist(discrete_path).max()

        if width <= 0:
            logger.debug(f"Width calculation failed for path: {discrete_path}")
            continue

        components.append((perimeter, width))

    return components


class ProfileCalculator(Analysis):
    """Calculates and analyzes 2D profile metrics for organelles."""

    def __init__(self, project: Project):
        super().__init__(project, ProfileProperties)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing the profile statistics.
        Refreshes the local stats view to include results calculated after initialization.
        """
        self.own_stats = [
            s for s in self.project.stats if isinstance(s.data, self.property_type)
        ]

        data_rows = []
        for stat in self.own_stats:
            row = {
                "ID": stat.meta.organelle_id,
                "axis": stat.meta.axis_used,
                "mean_perimeter": stat.data.mean_perimeter,
                "mean_width": stat.data.mean_width,
                "mean_ratio": stat.data.mean_ratio,
                "slice_count": len(stat.data.perimeters),
            }
            data_rows.append(row)

        return pd.DataFrame(data_rows)

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
            # computed is a list of lists: [[(p1, w1), (p2, w2)], [], [(p3, w3)], ...]
            # Each sub-list represents one slicing plane.
            computed = compute(*tasks)

            # Flatten the results: each slice_result is a list of (perimeter, width) tuples
            all_components = []
            for slice_result in computed:
                for component in slice_result:
                    all_components.append(component)

            # Filter pairs where the mesh was actually hit
            valid_results = [
                res for res in all_components if not np.isnan(res[0]) and res[0] > 0
            ]

            perimeters = [res[0] for res in valid_results]
            widths = [res[1] for res in valid_results]

            # Calculate ratio (width / perimeter) for each slice
            ratios = [w / p for w, p in zip(widths, perimeters)]

            attempted = num_slices if num_slices is not None else len(tasks)

            # Calculate means across slices
            m_perimeter = float(np.mean(perimeters)) if perimeters else np.nan
            m_width = float(np.mean(widths)) if widths else np.nan
            m_ratio = float(np.mean(ratios)) if ratios else np.nan

            data = ProfileProperties(
                perimeters=perimeters,
                widths=widths,
                ratios=ratios,
                mean_perimeter=m_perimeter,
                mean_width=m_width,
                mean_ratio=m_ratio,
            )
            meta = ProfileMeta(
                organelle_id=org_id,
                axis_used=axis_record,
                num_slices_attempted=attempted,
            )

            # Register the result in the central project registry
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
