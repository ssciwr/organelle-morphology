from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from organelle_morphology.analysis import Analysis
from organelle_morphology.records import PropertyBlock, Record

if TYPE_CHECKING:
    from organelle_morphology.project import Project

logger = logging.getLogger(__name__)


@dataclass
class CurvatureData(PropertyBlock):
    """Aggregated curvature statistics for a single organelle."""

    min_curvature: float
    max_curvature: float
    mean_curvature: float
    std_curvature: float
    median_curvature: float
    mean_absolute_curvature: float


@dataclass(frozen=True)
class CurvatureMetaData(PropertyBlock):
    """Context for the curvature calculation."""

    organelle_id: str
    curvature_radius: float
    num_vertices: int


class CurvatureAnalysis(Analysis):
    """Calculates and aggregates curvature statistics for organelles."""

    def __init__(self, project: Project):
        super().__init__(project, CurvatureData)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing the curvature statistics.
        Refreshes the local stats view to include results calculated after initialization.
        """
        if not self.own_records:
            columns = [f.name for f in fields(CurvatureData)] + ["num_vertices"]
            index = pd.MultiIndex.from_tuples(tuples=[], names=["Radius", "Organelle"])
            df = pd.DataFrame(index=index, columns=columns)
            return df

        data_rows = {}
        for stat in self.own_records:
            row = stat.data.to_dict()
            row["num_vertices"] = stat.meta.num_vertices
            data_rows[(stat.meta.curvature_radius, stat.meta.organelle_id)] = row

        df = pd.DataFrame(data_rows).T
        df.sort_index(inplace=True)
        df.index.names = ["Radius", "Organelle"]
        return df

    def calculate_curvature_stats(self, ids: str = "*") -> None:
        """
        Calculate curvature statistics for organelles across all sources.

        Iterates over each source, ensures curvature maps are computed,
        then aggregates per-organelle statistics and registers them.

        Args:
            ids: Glob-style filter for organelle ids. Default "*" includes all.
            recompute: If True, recalculate curvature maps even if they already exist.
        """
        # Group organelles by source to avoid redundant curvature calculations
        orgs_by_source: dict = {}
        for source in self.project.sources.values():
            orgs = source.get_organelles(ids=ids)
            if orgs:
                orgs_by_source[source] = orgs

        if not orgs_by_source:
            logger.warning(f"No organelles found matching ids='{ids}'")
            return

        total = sum(len(orgs) for orgs in orgs_by_source.values())
        self.project.logger.info(
            f"Computing curvature stats for {total} organelle(s) across "
            f"{len(orgs_by_source)} source(s)..."
        )

        for source, organelles in orgs_by_source.items():
            labels = [o.label for o in organelles]
            curvature_maps = source.calc_curvature(labels=labels)

            for org in organelles:
                curvature_map = curvature_maps.get(org.label)
                if curvature_map is None:
                    logger.warning(
                        f"No curvature map available for {org.id} (label {org.label}). Skipping."
                    )
                    continue

                # Compute aggregate statistics from the per-vertex curvature values
                flat = curvature_map.ravel()
                valid = flat[~np.isnan(flat)]

                if len(valid) == 0:
                    logger.warning(f"No valid curvature values for {org.id}. Skipping.")
                    continue

                data = CurvatureData(
                    min_curvature=float(np.min(valid)),
                    max_curvature=float(np.max(valid)),
                    mean_curvature=float(np.mean(valid)),
                    std_curvature=float(np.std(valid)),
                    median_curvature=float(np.median(valid)),
                    mean_absolute_curvature=float(np.mean(np.abs(valid))),
                )
                meta = CurvatureMetaData(
                    organelle_id=org.id,
                    curvature_radius=source.curvature_radius,
                    num_vertices=int(len(valid)),
                )

                record = Record(data=data, meta=meta, project=self.project)
                self.project.registry.add(record)

        self.project.logger.info(
            f"Curvature stats computed and registered for "
            f"{sum(len(orgs) for orgs in orgs_by_source.values())} organelle(s)."
        )
