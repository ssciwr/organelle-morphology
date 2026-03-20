from __future__ import annotations
from typing import TYPE_CHECKING, List, Any
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from organelle_morphology.project import Project
    from organelle_morphology.organelle import Organelle

class Statistics:
    """
    Calculates and aggregates morphological statistics from a Project.
    """

    def __init__(self, project: Project):
        self.project = project
    
    def get_mesh_properties(self) -> list[str]:
        """Returns a list of all available mesh properties."""
        mesh_properties = [
            "mesh_volume",
            "sphericity",
            "flatness_ratio",
            "water_tight",
            "mesh_area",
            "mesh_centroid",
            "mesh_inertia",
        ]
        return mesh_properties
    
    def get_skeleton_properties(self) -> list[str]:
        """Returns a list of all available skeleton properties."""
        skeleton_properties = [
            "num_nodes",
            "total_length",
            "std_length",
            "num_branch_points",
            "end_points",
            "mean_length",
            "longest_path",
            "mean_radius",
            "std_radius",
        ]
        return skeleton_properties
    
    def get_geometry_properties(self) -> list[str]:
        """Returns a list of all available geometry properties."""
        geometry_properties = [
            "voxel_solidity",
            "voxel_extent",
        ]
        return geometry_properties

    def get_contact_properties(self) -> list[str]:
        """Returns a list of all available contact properties."""
        contact_properties = [
            "n_contacts",
            "total_area",
            "mean_area",
            "std_area",
            "mean_dist",
            "std_dist",
        ]
        return contact_properties
    
    def get_properties(self) -> list[str]:
        """Returns a list of all available properties across mesh, skeleton, geometry, and contact sources."""
        all_properties = (
            self.get_mesh_properties() +
            self.get_skeleton_properties() +
            self.get_geometry_properties() +
            self.get_contact_properties()
        )
        return sorted(all_properties)

    def get_organelle_mesh_properties(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """
        Extracts only the requested mesh-based properties.
        """

        # Define the keys that only the mesh property provides
        mesh_keys = set(self.get_mesh_properties())
        to_extract = selected.intersection(mesh_keys)
        if not to_extract: return {}

        res = {}
        try:
            props = organelle.mesh_properties
            
            for p in selected:
                if p in props:
                    res[p] = props[p]
        except (KeyError, AttributeError, RuntimeError) as e:
            self.project.logger.warning(f"Failed to retrieve mesh properties for {organelle.id}: {e}")
            
        return res
    
    def get_skeleton_stats(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """Extracts requested graph/skeleton metrics."""
        skeleton_keys = set(self.get_skeleton_properties())
        to_extract = selected.intersection(skeleton_keys)
        
        # If no keys selected or skeleton hasn't been generated yet, skip
        if not to_extract or organelle.skeleton is None: return {}

        res = {}
        # Use skeleton_info directly
        info = organelle.skeleton_info
        for p in to_extract:
            if p in info:
                res[p] = info[p]
        return res

    def get_geometry_stats(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """Extracts requested voxel-based geometric data (solidity, extent, etc)."""

        voxel_keys = set(self.get_geometry_properties())  # things that require voxel data
        to_extract = selected.intersection(voxel_keys) # find intersection
        if not to_extract: return {} # return early if no voxel keys are requested

        res = {}
        try:
            # Assuming these keys exist in the organelle.geometric_data dictionary
            geo_data = organelle.geometric_data
            for p in selected:
                if p in geo_data:
                    val = geo_data[p]
                    if not hasattr(val, "compute"): # Only add if delayed / Dask array is computed
                        res[p] = val
        except (KeyError, AttributeError, RuntimeError) as e:
            self.project.logger.warning(f"Failed to retrieve geometry stats for {organelle.id}: {e}")
        return res
    
    def get_contact_stats(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """Extracts requested contact site (MCS) data."""
        contact_keys = set(self.get_contact_properties())
        to_extract = selected.intersection(contact_keys)
        if not to_extract:
            return {}

        mcs_data = organelle.mcs_dict
        if not mcs_data:
            return {}

        res = {}
        # Iterate through each MCS label (e.g., "0-0.01", "0-0.03") and its metrics
        for mcs_label, metrics in mcs_data.items():
            # For each requested base property (e.g., "n_contacts")
            for prop_name in to_extract:
                # If the metric exists for this MCS label, add it to the result
                # with a new key format: "MCS_LABEL-PROPERTY_NAME"
                if prop_name in metrics:
                    res[f"{mcs_label}-{prop_name}"] = metrics[prop_name]

        return res

    def get_dataframe(self, ids: str = "*", properties: List[str] = None) -> pd.DataFrame:
        """
        Returns a unified DataFrame with a user-defined selection of properties
        from mesh, skeleton, geometry, and contact sources.
        check get_properties to see all potentially available properties.

        :param ids: Glob-style filter for organelles (e.g., "mito_*").
        :param properties: List of keys to include. If None, defaults to 
                           ['mesh_volume', 'sphericity', 'flatness_ratio'].
        """

        if properties is None:
            properties = ["mesh_volume", "sphericity", "flatness_ratio"]
        
        selected_set = set(properties)

        try:
            organelles = self.project.get_organelles(ids=ids)
        except Exception as e:
            self.project.logger.error(f"Failed to retrieve organelles for ids '{ids}': {e}")
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

    def get_summary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates statistics. Booleans only contribute to the average/share."""
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