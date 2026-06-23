import pandas as pd

from dataclasses import asdict, fields
from organelle_morphology.analysis import Analysis
from organelle_morphology.source import SkeletonData, SkeletonMetaData


class Skeleton_Analysis(Analysis):
    """Methods to analyze the skeleton"""

    def __init__(self, project):
        super().__init__(project=project, property_type=SkeletonData)

    def get_dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame with skeleton properties and metadata.

        Returns:
            pd.DataFrame with skeleton data and metadata columns.
        """

        if not self.own_records:
            skeleton_props = [f.name for f in fields(SkeletonData)]
            meta_cols = [f.name for f in fields(SkeletonMetaData)]
            meta_cols = [
                f.name for f in fields(SkeletonMetaData) if f.name != "organelle_id"
            ]
            return pd.DataFrame(columns=["organelle_id"] + skeleton_props + meta_cols)

        rows = []
        for record in self.own_records:
            meta: SkeletonMetaData = record.meta
            data: SkeletonData = record.data
            row = {"organelle_id": meta.organelle_id}
            row.update(asdict(data))
            row.update(asdict(meta))
            rows.append(row)

        skeleton_props = [f.name for f in fields(SkeletonData)]
        meta_cols = [
            f.name for f in fields(SkeletonMetaData) if f.name != "organelle_id"
        ]
        all_cols = ["organelle_id"] + skeleton_props + meta_cols
        return pd.DataFrame(rows).reindex(columns=all_cols)
