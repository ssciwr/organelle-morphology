import pandas as pd
import numpy as np
from organelle_morphology.analysis import Analysis
from organelle_morphology.organelle import McsData
from typing import Optional

# Per-patch table contract. Patch geometry is in-memory only, so this frame is
# empty (but typed) after loading aggregate records from YAML.
PATCH_COLUMNS = [
    "mcs_label",
    "organelle",
    "partner",
    "patch_id",
    "area",
    "mean_dist",
    "min_dist",
    "max_dist",
    "truncated",
]
_PATCH_DTYPES = {
    "mcs_label": "object",
    "organelle": "object",
    "partner": "object",
    "patch_id": "object",
    "area": "float64",
    "mean_dist": "float64",
    "min_dist": "float64",
    "max_dist": "float64",
    "truncated": "bool",
}


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

    def _selected_records(self) -> list:
        """Records matching the configured MCS label filter."""
        return [
            stat
            for stat in self.own_records
            if not self.mcs_label_filter
            or stat.meta.mcs_label in self.mcs_label_filter
        ]

    def get_mcs_properties(self) -> pd.DataFrame:
        """The properties of the MCS between organelles
        Gathers data from all the organelles into one dataframe.

        One row per ``(mcs_label, organelle)`` aggregate; the columns are the
        ``McsData`` fields, including ``n_contacts`` (own patch count) and
        ``n_partners``.

        Returns:
            pd.DataFrame
        """

        if len(self.mcs_labels) == 0:
            raise RuntimeError("No mcs labels found, run a mcs search first!")

        records = self._selected_records()
        if len(records) == 0:
            raise RuntimeError(
                "No matching mcs labels found, adjust your filter settings!"
            )

        mcs_properties = {
            (stat.meta.mcs_label, stat.meta.organelle_id): stat.data.to_dict()
            for stat in records
        }

        mcs_df = pd.DataFrame(mcs_properties).T
        mcs_df.index = pd.MultiIndex.from_tuples(
            mcs_df.index, names=["mcs_label", "organelle"]
        )
        mcs_df.sort_index(inplace=True)

        return mcs_df

    def get_mcs_patches(self) -> pd.DataFrame:
        """Per-patch geometry for the current session.

        Flattens the in-memory ``Organelle.mcs`` patch lists into one row per
        patch, with the columns listed in :data:`PATCH_COLUMNS`. ``mean_dist``
        is area-weighted over the patch's samples. Patch geometry is never
        persisted, so after ``load_all_from_yaml`` (or before any MCS search)
        this returns an empty DataFrame with the same typed columns.

        Filters: ``self.ids`` for organelles and ``self.mcs_label_filter`` for
        labels.

        Returns:
            pd.DataFrame: One row per in-memory patch, columns
                :data:`PATCH_COLUMNS`.
        """
        rows = []
        for organelle in self.project.get_organelles(ids=self.ids):
            for label, partners in organelle.mcs.items():
                if self.mcs_label_filter and label not in self.mcs_label_filter:
                    continue
                for partner, patches in partners.items():
                    for patch in patches:
                        distances = np.asarray(
                            patch["distances"], dtype=np.float64
                        )
                        weights = patch.get("sample_areas")
                        if weights is not None:
                            weights = np.asarray(weights, dtype=np.float64)
                        if distances.size:
                            if (
                                weights is not None
                                and weights.shape == distances.shape
                                and float(weights.sum()) > 0.0
                            ):
                                mean_dist = float(
                                    np.average(distances, weights=weights)
                                )
                            else:
                                mean_dist = float(np.mean(distances))
                            min_dist = float(np.min(distances))
                            max_dist = float(np.max(distances))
                        else:
                            mean_dist = min_dist = max_dist = np.nan
                        rows.append(
                            {
                                "mcs_label": label,
                                "organelle": organelle.id,
                                "partner": partner,
                                "patch_id": patch["patch_id"],
                                "area": float(patch["area"]),
                                "mean_dist": mean_dist,
                                "min_dist": min_dist,
                                "max_dist": max_dist,
                                "truncated": bool(patch["truncated"]),
                            }
                        )

        if not rows:
            return pd.DataFrame(
                {
                    column: pd.Series(dtype=dtype)
                    for column, dtype in _PATCH_DTYPES.items()
                }
            )

        return pd.DataFrame(rows, columns=PATCH_COLUMNS).astype(_PATCH_DTYPES)

    def get_mcs_overview(self):
        """Aggregate per-label overview.

        Weights are own patch counts (``McsData.n_contacts``). ``total_pairs``
        counts unordered organelle edges (each reciprocal pair once) and does
        not require the partner to have its own record.

        Returns:
            pd.DataFrame: Overview with a ``(group, measure)`` MultiIndex and
                one column per MCS label.
        """

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

            # Deduplicate organelle pairs without mutating record data. A pair
            # is an unordered edge; partners without their own record (e.g.
            # filtered out) still form an edge instead of raising.
            organelle_ids = x.index.get_level_values(1)
            edges = {
                frozenset((organelle, partner))
                for organelle, partners in zip(organelle_ids, x["partners"])
                for partner in partners
            }
            total_pairs = len(edges)
            total_partners = x["n_partners"].sum()

            new_columns = [
                ("overall", "total_pairs"),
                ("overall", "total_sites"),
                ("overall", "total_area"),
                ("overall", "total_partners"),
                ("per organelle", "mean_contacts"),
                ("per organelle", "std_contacts"),
                ("per organelle", "mean_partners"),
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
                    total_pairs,
                    total_contacts,
                    total_area,
                    total_partners,
                    mean_n_contacts,
                    std_n_contacts,
                    x["n_partners"].mean(),
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
