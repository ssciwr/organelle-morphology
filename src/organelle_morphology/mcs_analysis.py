import pandas as pd
import numpy as np
from organelle_morphology.analysis import Analysis
from organelle_morphology.organelle import McsData
from typing import Optional


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
