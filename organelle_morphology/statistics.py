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

    def get_organelle_mesh_properties(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """
        Extracts only the requested mesh-based properties.
        """

        # Define the keys that only the mesh property provides
        mesh_keys = {"mesh_volume", "mesh_area", "mesh_centroid", "mesh_inertia", 
                     "water_tight", "sphericity", "flatness_ratio"}
        to_extract = selected.intersection(mesh_keys)
        if not to_extract: return {}

        res = {}
        try:
            # We only access .mesh_properties if at least one mesh property is requested
            # as this call often triggers the expensive mesh computation.
            props = organelle.mesh_properties
            
            for p in selected:
                if p in props:
                    res[p] = props[p]
        except Exception as e:
            res["mesh_error"] = str(e)
            
        return res
    
    def get_skeleton_stats(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """Extracts requested graph/skeleton metrics."""
        skeleton_keys = {"num_nodes", "num_edges", "total_length", "avg_edge_length"}
        to_extract = selected.intersection(skeleton_keys)
        
        # If no keys selected OR skeleton hasn't been generated yet, skip
        if not to_extract or organelle.skeleton is None:
            return {}

        res = {}
        # Use skeleton_info directly
        info = organelle.skeleton_info
        for p in to_extract:
            if p in info:
                res[p] = info[p]
        return res

    def get_geometry_stats(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """Extracts requested voxel-based geometric data (solidity, extent, etc)."""

        voxel_keys = {"solidity", "extent"}  # things that require voxel data
        to_extract = selected.intersection(voxel_keys) # find intersection
        if not to_extract: return {} # return early if no voxel keys are requested

        res = {}
        try:
            # Only trigger if voxel keys are requested
            # Assuming these keys exist in the organelle.geometric_data dictionary
            geo_data = organelle.geometric_data
            for p in selected:
                if p in geo_data:
                    res[p] = geo_data[p]
        except Exception:
            pass
        return res

    def get_contact_stats(self, organelle: Organelle, selected: set[str]) -> dict[str, Any]:
        """Extracts requested contact site (MCS) data."""
        # Use the specific keys selected by the user
        mcs_keys = {k for k in selected if k.startswith("contact_")}
        if not mcs_keys: return {}

        res = {}
        mcs_data = organelle.mcs_dict
        for p in mcs_keys:
            if p in mcs_data:
                res[p] = mcs_data[p]
        return res

    def get_dataframe(self, ids: str = "*", properties: List[str] = None) -> pd.DataFrame:
        """
        Returns a unified DataFrame with a user-defined selection of properties
        from mesh, skeleton, geometry, and contact sources.

        Available keys:
        - Mesh: 'mesh_volume', 'mesh_area', 'mesh_centroid', 'mesh_inertia', 
                'water_tight', 'sphericity', 'flatness_ratio'
        - Skeleton: 'num_nodes', 'num_edges', 'total_length', 'avg_edge_length'
        - Geometry (Voxel): 'solidity', 'extent'
        - Contact: 'contact_er', 'contact_mito' (or other 'contact_*' keys)

        :param ids: Glob-style filter for organelles (e.g., "mito_*").
        :param properties: List of keys to include. If None, defaults to 
                           ['mesh_volume', 'sphericity', 'flatness_ratio'].
        """

        if properties is None:
            properties = ["mesh_volume", "sphericity", "flatness_ratio"]
        
        selected_set = set(properties)

        try:
            organelles = self.project.get_organelles(ids=ids)
        except Exception:
            # If retrieval fails, return empty DF
            return pd.DataFrame()

        if not organelles:
            return pd.DataFrame()

        # 2. Collect Data
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

        for prop in selected_set:
            if prop not in df.columns:
                df[prop] = np.nan

        cols = sorted([c for c in df.columns if c != "ID"])
        return df[["ID"] + cols]

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