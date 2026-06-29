from __future__ import annotations

from dataclasses import fields, asdict
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd

from organelle_morphology.organelle import McsData
from organelle_morphology.records import PropertyBlock

if TYPE_CHECKING:
    from organelle_morphology.organelle import Organelle
    from organelle_morphology.project import Project

logger = logging.getLogger(__name__)


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

        return summary_df


class Mcs_Analysis(Analysis):
    """Methods to calculate mcs statistics"""

    def __init__(self, project):
        super().__init__(project=project, property_type=McsData)

    def __post_init__(self):
        self.set_filters()

    def set_filters(self, ids: str = "*", mcs_labels: Optional[list[str]] = None):
        """Filter records for everything in this analysis.

        Args:
            ids: Glob-style filter pattern for organelles, defaults to "*"
            mcs_labels: Filter the MCS calculations, defaults to None
        """
        self.ids = ids
        self.mcs_label_filter = mcs_labels

    @property
    def mcs_labels(self) -> list:
        return list({s.meta.mcs_label for s in self.own_records})

    def get_mcs_properties(self) -> pd.DataFrame:
        """The properties of the MCS between organelles
        Gathers data from all the organelles into one dataframe.


        Returns:
            pd.DataFrame
        """

        if len(self.mcs_labels) == 0:
            raise RuntimeError("No mcs labels found, run a mcs search first!")

        mcs_properties = {}

        for stat in self.own_records:
            for label in self.mcs_labels:
                if self.mcs_label_filter and (label not in self.mcs_label_filter):
                    continue
                elif label == stat.meta.mcs_label:
                    mcs_properties[(label, stat.meta.organelle_id)] = (
                        stat.data.to_dict()
                    )

        if len(mcs_properties) == 0:
            raise RuntimeError(
                "No matching mcs labels found, adjust your filter settings!"
            )

        mcs_df = pd.DataFrame(mcs_properties).T
        mcs_df.sort_index(inplace=True)
        mcs_df.index.names = ["mcs_label", "organelle"]

        return mcs_df

    def get_mcs_overview(self):
        def _weighted_stats(x):
            # Calculate the weighted mean and standard deviation for 'mean_area' and 'mean_dist'
            mean_area = np.average(x["mean_area"], weights=x["n_contacts"])
            std_area = np.sqrt(
                np.average(
                    (x["std_area"] ** 2 + (x["mean_area"] - mean_area) ** 2),
                    weights=x["n_contacts"],
                )
            )
            mean_dist = np.average(x["mean_dist"], weights=x["n_contacts"])
            std_dist = np.sqrt(
                np.average(
                    (x["std_dist"] ** 2 + (x["mean_dist"] - mean_dist) ** 2),
                    weights=x["n_contacts"],
                )
            )
            mean_min_dist_mcs = np.average(x["min_dist"], weights=x["n_contacts"])

            # Calculate per organelle
            total_contacts = x["n_contacts"].sum()
            mean_n_contacts = x["n_contacts"].mean()
            std_n_contacts = x["n_contacts"].std()
            total_area = x["total_area"].sum()
            mean_total_area = x["total_area"].mean()
            std_total_area = x["total_area"].std()

            mean_min_dist = x["min_dist"].mean()
            mean_contacts_per_vol = x["n_contacts_per_volume"].mean()
            mean_contacts_per_area = x["n_contacts_per_area"].mean()

            new_columns = [
                ("overall", "total_contacts"),
                ("overall", "total_area"),
                ("per organelle", "mean_contacts"),
                ("per organelle", "std_contacts"),
                ("per organelle", "mean_total_area"),
                ("per organelle", "std_area"),
                ("per organelle", "mean_contacts_per_volume"),
                ("per organelle", "mean_contacts_per_area"),
                ("per organelle", "mean_min_dist"),
                ("per mcs", "mean_area"),
                ("per mcs", "std_area"),
                ("per mcs", "mean_dist"),
                ("per mcs", "std_dist"),
                ("per mcs", "mean_min_dist"),
            ]
            new_index = pd.MultiIndex.from_tuples(new_columns)

            return pd.Series(
                [
                    total_contacts,
                    total_area,
                    mean_n_contacts,
                    std_n_contacts,
                    mean_total_area,
                    std_total_area,
                    mean_contacts_per_vol,
                    mean_contacts_per_area,
                    mean_min_dist,
                    mean_area,
                    std_area,
                    mean_dist,
                    std_dist,
                    mean_min_dist_mcs,
                ],
                index=new_index,
            )

        mcs_df = self.get_mcs_properties()

        overview = mcs_df.groupby(level=0).apply(_weighted_stats)
        overview.sort_index(axis=1, inplace=True)
        return overview.T

    def get_dataframe(self) -> pd.DataFrame:
        return self.get_mcs_properties()


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
            "total_area",
            "mean_area",
            "std_area",
            "mean_dist",
            "std_dist",
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
    ) -> dict[str, Any]:
        """Extracts requested contact site (MCS) data."""
        contact_keys = set(self.get_contact_properties())
        to_extract = selected.intersection(contact_keys)
        if not to_extract:
            return {}

        res = {}
        for stat in self.own_records:
            if stat.meta.organelle_id == organelle.id:
                mcs_label = stat.meta.mcs_label
                # For each requested base property (e.g., "n_contacts")
                for prop_name in to_extract:
                    # If the metric exists for this MCS label, add it to the result
                    # with a new key format: "MCS_LABEL-PROPERTY_NAME"
                    if hasattr(stat.data, prop_name):
                        res[f"{mcs_label}-{prop_name}"] = getattr(stat.data, prop_name)

        return res

    def get_dataframe(
        self, ids: str = "*", properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Returns a unified DataFrame with a user-defined selection of properties
        from mesh, skeleton, geometry, and contact sources.
        check get_properties to see all potentially available properties.

        :param ids: Glob-style filter for organelles (e.g., "mito_*").
        :param properties: List of keys to include. If None, defaults to
                           ['volume', 'sphericity', 'flatness_ratio'].
        """

        if properties is None:
            properties = ["volume", "sphericity", "flatness_ratio"]

        selected_set = set(properties)

        try:
            organelles = self.project.get_organelles(ids=ids)
        except Exception as e:
            self.project.logger.error(
                f"Failed to retrieve organelles for ids '{ids}': {e}"
            )
            # If retrieval fails, return empty DF with correct columns
            return pd.DataFrame(columns=["ID"] + sorted(list(selected_set)))

        if not organelles:
            return pd.DataFrame(columns=["ID"] + sorted(list(selected_set)))

        # Collect Data
        data_rows = []
        for org in organelles:
            row = {"ID": org.id}

            # Aggregate properties from all specialized helper methods
            row.update(self.get_organelle_mesh_properties(org, selected_set))
            row.update(self.get_skeleton_stats(org, selected_set))
            row.update(self.get_geometry_stats(org, selected_set))
            row.update(self.get_contact_stats(org, selected_set))

            data_rows.append(row)

        df = pd.DataFrame(data_rows)

        # Determine Final Columns and handle dynamic contact columns (e.g., "0-0.01-n_contacts")
        # which correspond to the generic selected keys (e.g., "n_contacts").
        contact_props = set(self.get_contact_properties())
        selected_contact_props = selected_set.intersection(contact_props)
        static_props = selected_set - contact_props

        final_cols = ["ID"] + sorted(list(static_props))

        if not df.empty:
            dynamic_cols = []
            for col in df.columns:
                # Check if the column ends with a selected contact property
                parts = col.rsplit("-", 1)
                if len(parts) == 2 and parts[1] in selected_contact_props:
                    dynamic_cols.append(col)
            final_cols.extend(sorted(dynamic_cols))

        return df.reindex(columns=final_cols)
