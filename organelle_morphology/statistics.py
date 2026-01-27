from __future__ import annotations
from typing import TYPE_CHECKING, List
import pandas as pd # <--- Added dependency

if TYPE_CHECKING:
    from organelle_morphology.project import Project

class Statistics:
    """
    Calculates and aggregates morphological statistics from a Project.
    """

    def __init__(self, project: Project):
        self.project = project

    def get_dataframe(self, ids: str = "*") -> pd.DataFrame:
        """
        Returns a Pandas DataFrame containing Volume, Sphericity, and Flatness 
        for organelles matching the filter.
        """
        # 1. Retrieval
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
            try:
                # Trigger computation
                props = org.mesh_properties

                # Safe Extraction
                row = {
                    "ID": str(org.id),
                    "Volume (vx)": float(props.get('mesh_volume', 0.0)),
                    "Sphericity": float(props.get('sphericity', 0.0)),
                    "Flatness": float(props.get('flatness_ratio', 0.0))
                }
                data_rows.append(row)
                
            except Exception as e:
                # Log error in the row
                data_rows.append({
                    "ID": str(org.id),
                    "Volume (vx)": -1.0, 
                    "Sphericity": -1.0,
                    "Flatness": -1.0,
                    "Error": str(e)
                })

        # 3. Create DataFrame
        df = pd.DataFrame(data_rows)
        return df