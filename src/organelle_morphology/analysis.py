from __future__ import annotations

from dataclasses import fields, asdict
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd

from organelle_morphology.records import PropertyBlock

if TYPE_CHECKING:
    from organelle_morphology.organelle import Organelle
    from organelle_morphology.project import Project

logger = logging.getLogger(__name__)

# Column group used for columns that are not MCS properties in the structured
# DataFrames built by ``Misc_Analysis.get_dataframe``. MCS columns use their
# full label as the group, so a label containing "-" (e.g. "0-0.1,mito-er")
# stays intact.
STATIC_COLUMN_GROUP = "static"


def _static_columns(values: dict[str, Any]) -> dict[tuple[str, str], Any]:
    """Prefix non-MCS property names with :data:`STATIC_COLUMN_GROUP`."""
    return {(STATIC_COLUMN_GROUP, key): value for key, value in values.items()}


def column_display_name(column) -> str:
    """Human-readable label for a column of ``Misc_Analysis.get_dataframe``.

    Structured columns are ``(group, property)`` tuples; non-MCS columns use
    :data:`STATIC_COLUMN_GROUP` and render as the bare property name, MCS
    columns as ``"<mcs_label> | <property>"``. Plain labels (for example
    ``"Measure"``) are returned unchanged as strings.
    """
    if isinstance(column, tuple) and len(column) == 2:
        group, prop = column
        if group == STATIC_COLUMN_GROUP:
            return str(prop)
        return f"{group} | {prop}"
    return str(column)



class Analysis(ABC):
    """Analysis base class. Specific analysis workflows should subclass this."""

    def __init__(self, project: Project, property_type: type[PropertyBlock]):
        """Instantiate new Analysis

        Args:
            project: The project
            property_type: The PropertyBlock class of the data (not meta data)
        """
        self.project = project
        self.property_type = property_type

        self.__post_init__()

    def __post_init__(self):
        """Run after initialization of base class, used for subclass specific init"""
        pass

    @property
    def own_records(self):
        return self.project.registry.get_by_type(self.property_type)

    def save_records(self):
        self.project.registry.save_all_to_yaml()

    def clean_own_records(self):
        for record in self.own_records:
            self.project.registry.clear_record(record)

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_summary_dataframe(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculates statistics. Booleans only contribute to the average/share."""
        if df is None:  # if no filtered dataframe provided, use the full one
            df = self.get_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = df.select_dtypes(include=[bool]).columns.tolist()

        if not (numeric_cols or bool_cols):
            return pd.DataFrame()

        calc_df = df.copy()
        for col in bool_cols:
            calc_df[col] = calc_df[col].astype(float)

        stats_rows = {}
        stats_rows["Average (or Share)"] = calc_df[numeric_cols + bool_cols].mean()

        numeric_df = calc_df[numeric_cols]
        if not numeric_df.empty:
            stats_rows["Std Dev"] = numeric_df.std()
            stats_rows["Median (50th percentile)"] = numeric_df.median()
            stats_rows["Minimum"] = numeric_df.min()
            stats_rows["Maximum"] = numeric_df.max()
            stats_rows["16th percentile"] = numeric_df.quantile(0.16)
            stats_rows["84th percentile"] = numeric_df.quantile(0.84)

            stats_rows["Geometric Mean"] = numeric_df.apply(
                lambda x: np.exp(np.log(x[x > 0]).mean()) if np.any(x > 0) else np.nan
            )

        summary_df = pd.DataFrame(stats_rows).T
        target_order = [c for c in df.columns if c in summary_df.columns]
        summary_df = summary_df.reindex(columns=target_order)
        summary_df.index.name = "Measure"
        summary_df = summary_df.reset_index()
        if isinstance(summary_df.columns, pd.MultiIndex):
            # A structured (group, property) frame keeps its tuples, but the
            # inserted "Measure" index column would otherwise become
            # ("Measure", ""). Collapse that one level back to a plain label.
            summary_df.columns = pd.Index(
                [
                    "Measure" if (column[1] == "" and column[0] == "Measure") else column
                    for column in summary_df.columns
                ]
            )

        return summary_df


# should be split into separate topics, like mcs, skeleton, misc(?)
class Misc_Analysis(Analysis):
    """
    Calculates and aggregates morphological statistics from a Project.
    """

    def get_mesh_properties(self) -> list[str]:
        """Returns a list of all available mesh properties."""
        mesh_properties = [
            "volume",
            "sphericity",
            "flatness_ratio",
            "water_tight",
            "area",
            "centroid",
            "inertia",
        ]
        return mesh_properties

    def get_skeleton_properties(self) -> list[str]:
        """Returns a list of all available skeleton properties."""
        from organelle_morphology.skeleton_analysis import SkeletonData

        return [f.name for f in fields(SkeletonData)]

    def get_geometry_properties(self) -> list[str]:
        """Returns a list of all available geometry properties."""
        geometry_properties = [
            "voxel_solidity",
            "voxel_extent",
        ]
        return geometry_properties

    def get_contact_properties(self) -> list[str]:
        """Returns a list of all available possible MCS properties
        calculated on each Organelle.
        """
        contact_properties = [
            "n_contacts",
            "n_partners",
            "total_area",
            "mean_area",
            "std_area",
            "mean_dist",
            "std_dist",
            "min_dist",
            "max_dist",
            "n_contacts_per_area",
            "n_contacts_per_volume",
            "area_per_area",
            "area_per_volume",
        ]
        return contact_properties

    def get_organelle_mesh_properties(
        self, organelle: Organelle, selected: set[str]
    ) -> dict[str, Any]:
        """
        Extracts only the requested mesh-based properties.
        """

        # Define the keys that only the mesh property provides
        mesh_keys = set(self.get_mesh_properties())
        to_extract = selected.intersection(mesh_keys)
        if not to_extract:
            return {}

        res = {}
        try:
            props = organelle.mesh_properties

            for p in selected:
                if hasattr(props, p):
                    res[p] = getattr(props, p)
        except (KeyError, AttributeError, RuntimeError) as e:
            self.project.logger.warning(
                f"Failed to retrieve mesh properties for {organelle.id}: {e}"
            )

        return res

    def get_skeleton_stats(
        self, organelle: Organelle, selected: set[str]
    ) -> dict[str, Any]:
        """Extracts requested graph/skeleton metrics."""
        skeleton_keys = set(self.get_skeleton_properties())
        to_extract = selected.intersection(skeleton_keys)

        # If no skeleton keys are requested, skip the loop entirely
        if not to_extract:
            return {}

        res = {}
        # Use skeleton_info directly
        info = organelle.skeleton_info
        if info:
            for p in to_extract:
                if p in asdict(info):
                    res[p] = getattr(info, p)
        return res

    def get_geometry_stats(
        self, organelle: Organelle, selected: set[str]
    ) -> dict[str, Any]:
        """Extracts requested voxel-based geometric data (solidity, extent, etc)."""

        voxel_keys = set(
            self.get_geometry_properties()
        )  # things that require voxel data
        to_extract = selected.intersection(voxel_keys)  # find intersection
        if not to_extract:
            return {}  # return early if no voxel keys are requested

        res = {}
        try:
            # Assuming these keys exist in the organelle.geometric_data dictionary
            geo_data = organelle.geometric_data
            for p in selected:
                if p in geo_data:
                    val = geo_data[p]
                    if hasattr(val, "compute"):  # If it's a Dask/Delayed object,
                        res[p] = val.compute()  # compute it
                    else:
                        res[p] = val  # otherwise, take it as is
        except (KeyError, AttributeError, RuntimeError) as e:
            self.project.logger.warning(
                f"Failed to retrieve geometry stats for {organelle.id}: {e}"
            )
        return res

    def get_contact_stats(
        self, organelle: Organelle, selected: set[str]
    ) -> dict[tuple[str, str], Any]:
        """Extract the requested MCS properties for one organelle.

        Args:
            organelle: Organelle whose aggregate MCS records are read.
            selected: Requested property names; MCS property names are
                extracted, everything else is ignored.

        Returns:
            dict[tuple[str, str], Any]: Mapping keyed by
            ``(mcs_label, property)``. The label is kept whole (it may
            contain ``-``, e.g. ``"0-0.1,mito-er"``), so callers never have
            to split a concatenated column name.
        """
        contact_keys = set(self.get_contact_properties())
        to_extract = selected.intersection(contact_keys)
        if not to_extract:
            return {}

        res = {}
        for stat in self.own_records:
            if not hasattr(stat.meta, "mcs_label"):
                continue
            if stat.meta.organelle_id == organelle.id:
                mcs_label = stat.meta.mcs_label
                for prop_name in to_extract:
                    if hasattr(stat.data, prop_name):
                        res[(mcs_label, prop_name)] = getattr(
                            stat.data, prop_name
                        )

        return res

    def get_dataframe(
        self, ids: str = "*", properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Returns a unified DataFrame with a user-defined selection of properties
        from mesh, skeleton, geometry, and MCS sources.
        check get_properties to see all potentially available properties.

        Column contract: a 2-level column MultiIndex with names
        ``("group", "property")``. Non-MCS columns use
        :data:`STATIC_COLUMN_GROUP` (``"static"``) as the group; MCS
        columns use the full MCS label as the group, for example
        ``("0-0.1,mito-er", "n_contacts")``. This keeps labels containing
        ``"-"`` intact and removes any string parsing of column names.

        :param ids: Glob-style filter for organelles (e.g., "mito_*").
        :param properties: List of keys to include. If None, defaults to
                           ['volume', 'sphericity', 'flatness_ratio'].
        """

        if properties is None:
            properties = ["volume", "sphericity", "flatness_ratio"]

        selected_set = set(properties)
        contact_props = set(self.get_contact_properties())
        selected_contact_props = selected_set.intersection(contact_props)
        static_props = selected_set - contact_props

        def _empty_frame() -> pd.DataFrame:
            columns = [(STATIC_COLUMN_GROUP, "ID")] + [
                (STATIC_COLUMN_GROUP, prop) for prop in sorted(static_props)
            ]
            return pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(
                    columns, names=["group", "property"]
                )
            )

        try:
            organelles = self.project.get_organelles(ids=ids)
        except Exception as e:
            self.project.logger.error(
                f"Failed to retrieve organelles for ids '{ids}': {e}"
            )
            # If retrieval fails, return empty DF with correct columns
            return _empty_frame()

        if not organelles:
            return _empty_frame()

        # Collect Data. Every column key is already a structured
        # (group, property) tuple, so no column name parsing is needed below.
        data_rows = []
        for org in organelles:
            row = {(STATIC_COLUMN_GROUP, "ID"): org.id}

            # Aggregate properties from all specialized helper methods
            row.update(
                _static_columns(
                    self.get_organelle_mesh_properties(org, selected_set)
                )
            )
            row.update(
                _static_columns(self.get_skeleton_stats(org, selected_set))
            )
            row.update(
                _static_columns(self.get_geometry_stats(org, selected_set))
            )
            row.update(self.get_contact_stats(org, selected_set))

            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df.columns = pd.MultiIndex.from_tuples(
            df.columns, names=["group", "property"]
        )

        final_cols = [(STATIC_COLUMN_GROUP, "ID")] + [
            (STATIC_COLUMN_GROUP, prop) for prop in sorted(static_props)
        ]
        dynamic_cols = sorted(
            column
            for column in df.columns
            if column[0] != STATIC_COLUMN_GROUP
            and column[1] in selected_contact_props
        )
        final_cols.extend(dynamic_cols)

        return df.reindex(
            columns=pd.MultiIndex.from_tuples(
                final_cols, names=["group", "property"]
            )
        )
