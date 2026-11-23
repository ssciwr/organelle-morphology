import pytest
import pandas as pd
import numpy as np

from organelle_morphology.mcs_analysis import Mcs_Analysis
from organelle_morphology.organelle import (
    CURRENT_MCS_SCHEMA_VERSION,
    McsData,
    McsMetadata,
)
from organelle_morphology.records import Record

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


def _patch(patch_id, area, distances, truncated=False, sample_areas=None):
    distances = np.asarray(distances, dtype=np.float64)
    if sample_areas is None:
        sample_areas = np.full(len(distances), float(area) / len(distances))
    return {
        "patch_id": patch_id,
        "faces_index": np.arange(len(distances), dtype=np.int64),
        "distances": distances,
        "sample_areas": np.asarray(sample_areas, dtype=np.float64),
        "area": float(area),
        "truncated": truncated,
        "partner_patch_ids": (),
    }


def _mcs_dict(label, self_id, partner_id, patches):
    return {
        "self_id": self_id,
        "partner_id": partner_id,
        "mcs_label": label,
        "patches": patches,
    }


def _mcs_meta(label, min_distance=0.0, max_distance=0.1):
    return {
        "ids_filter_1": "*",
        "ids_filter_2": "*",
        "min_distance": min_distance,
        "max_distance": max_distance,
    }


def _add_mcs_record(
    project,
    organelle_id,
    label,
    partners,
    n_contacts,
    mean_area=10.0,
    mean_dist=0.05,
):
    """Add a hand-built aggregate record (no patch geometry needed)."""
    meta = McsMetadata(
        mcs_label=label,
        organelle_id=organelle_id,
        max_dist=0.1,
        schema_version=CURRENT_MCS_SCHEMA_VERSION,
    )
    data = McsData(
        n_contacts=n_contacts,
        n_partners=len(partners),
        total_area=mean_area * n_contacts,
        mean_area=mean_area,
        std_area=0.0,
        mean_dist=mean_dist,
        std_dist=0.0,
        min_dist=mean_dist,
        max_dist=mean_dist,
        partners=tuple(partners),
        min_dist_per_org=tuple(mean_dist for _ in partners),
        max_dist_per_org=tuple(mean_dist for _ in partners),
        n_contacts_per_area=0.0,
        n_contacts_per_volume=0.0,
        area_per_area=0.0,
        area_per_volume=0.0,
    )
    record = Record(data=data, meta=meta, project=project)
    project.registry.add(record)
    return record


def test_init_defaults(project_with_sources):
    """Test that Mcs_Analysis initializes with correct defaults."""
    analysis = Mcs_Analysis(project_with_sources)
    assert analysis.project is project_with_sources
    assert analysis.ids == "*"
    assert analysis.mcs_label_filter is None

    analysis.set_filters(ids="mito_0001")
    assert analysis.ids == "mito_0001"

    analysis.set_filters(mcs_labels=["0.0-10.0,mito-mito"])
    assert analysis.mcs_label_filter == ["0.0-10.0,mito-mito"]

    analysis.set_filters(ids="mito_*", mcs_labels=["0.0-10.0,mito-mito"])
    assert analysis.ids == "mito_*"
    assert analysis.mcs_label_filter == ["0.0-10.0,mito-mito"]


def test_mcs_labels_empty(project_with_sources):
    """Test that mcs_labels returns empty list when no MCS data exists."""
    analysis = Mcs_Analysis(project_with_sources)
    assert analysis.mcs_labels == []

    with pytest.raises(RuntimeError, match="No mcs labels found"):
        analysis.get_mcs_properties()
    with pytest.raises(RuntimeError, match="No mcs labels found"):
        analysis.get_mcs_overview()


def test_get_mcs_patches_one_row_per_patch(project_with_sources):
    """Patch table flattens every in-memory patch of a pair."""
    org = project_with_sources.get_organelles("mito_0015")[0]
    label = "0.0-0.1,mito-er"
    meta = _mcs_meta(label)
    partner_a = "er_9001"
    partner_b = "er_9002"

    org.add_mcs(
        _mcs_dict(
            label,
            org.id,
            partner_a,
            [
                _patch("a~0", 4.0, [0.02, 0.04, 0.06, 0.08], truncated=True),
                _patch("a~1", 1.0, [0.01, 0.03]),
            ],
        ),
        meta,
    )
    org.add_mcs(
        _mcs_dict(label, org.id, partner_b, [_patch("b~0", 2.0, [0.09])]),
        meta,
    )

    analysis = Mcs_Analysis(project_with_sources)
    analysis.set_filters(mcs_labels=[label])
    df = analysis.get_mcs_patches()

    assert list(df.columns) == PATCH_COLUMNS
    assert len(df) == 3
    assert (df["organelle"] == org.id).all()
    assert (df["mcs_label"] == label).all()
    assert df["partner"].value_counts().to_dict() == {partner_a: 2, partner_b: 1}

    first = df[df["patch_id"] == "a~0"].iloc[0]
    assert first["area"] == pytest.approx(4.0)
    assert first["mean_dist"] == pytest.approx(0.05)
    assert first["min_dist"] == pytest.approx(0.02)
    assert first["max_dist"] == pytest.approx(0.08)
    assert bool(first["truncated"]) is True

    second = df[df["patch_id"] == "a~1"].iloc[0]
    assert bool(second["truncated"]) is False


def test_get_mcs_patches_mean_dist_is_area_weighted(project_with_sources):
    """Per-patch distance means use the sample areas, not equal weights."""
    org = project_with_sources.get_organelles("mito_0015")[0]
    label = "0.0-0.1,mito-er"
    meta = _mcs_meta(label)
    org.add_mcs(
        _mcs_dict(
            label,
            org.id,
            "er_9001",
            [
                _patch(
                    "w~0",
                    10.0,
                    [0.02, 0.08],
                    sample_areas=[9.0, 1.0],
                )
            ],
        ),
        meta,
    )

    analysis = Mcs_Analysis(project_with_sources)
    analysis.set_filters(mcs_labels=[label])
    df = analysis.get_mcs_patches()

    assert df.iloc[0]["mean_dist"] == pytest.approx(0.026)
    assert df.iloc[0]["min_dist"] == pytest.approx(0.02)
    assert df.iloc[0]["max_dist"] == pytest.approx(0.08)


def test_get_mcs_patches_empty_is_typed(project_with_sources):
    """Without in-memory patch geometry the table is empty but typed."""
    analysis = Mcs_Analysis(project_with_sources)
    df = analysis.get_mcs_patches()

    assert df.empty
    assert list(df.columns) == PATCH_COLUMNS
    assert df["area"].dtype == np.dtype("float64")
    assert df["mean_dist"].dtype == np.dtype("float64")
    assert df["truncated"].dtype == np.dtype("bool")


def test_get_mcs_overview_dedups_pairs_without_mutating_records(
    project_with_sources,
):
    """Reciprocal partners count once; partners without a record don't raise."""
    label = "0.0-0.1,mito-er"
    record_a = _add_mcs_record(
        project_with_sources, "mito_0001", label, ["er_0001"], 2, mean_area=10.0
    )
    record_b = _add_mcs_record(
        project_with_sources, "er_0001", label, ["mito_0001"], 3, mean_area=20.0
    )
    # er_9999 has no record for this label: the old orgs.index lookup raised.
    record_c = _add_mcs_record(
        project_with_sources, "mito_0002", label, ["er_9999"], 1, mean_area=30.0
    )

    before = {
        record.meta.organelle_id: tuple(record.data.partners)
        for record in (record_a, record_b, record_c)
    }
    analysis = Mcs_Analysis(project_with_sources)
    analysis.set_filters(mcs_labels=[label])
    overview = analysis.get_mcs_overview()

    assert overview.loc[("overall", "total_pairs")][label] == 2
    assert overview.loc[("overall", "total_sites")][label] == 6
    # Weighted by patch counts (2, 3, 1), not by organelle.
    assert overview.loc[("per mcs", "mean_area")][label] == pytest.approx(
        (2 * 10.0 + 3 * 20.0 + 1 * 30.0) / 6.0
    )

    after = {
        record.meta.organelle_id: tuple(record.data.partners)
        for record in (record_a, record_b, record_c)
    }
    assert after == before


class TestMcsAnalysisWithData:
    """Tests for Mcs_Analysis behavior after MCS data has been generated."""
    @pytest.fixture
    def analysis_with_mcs(self, project_with_sources):
        """Fixture providing an Mcs_Analysis with MCS data."""
        project_with_sources.search_mcs(max_distance=10.0, min_distance=0.0)
        analysis = Mcs_Analysis(project_with_sources)
        return analysis

    def test_get_mcs_properties_contains_expected_columns(self, analysis_with_mcs):
        # Test labels
        labels = analysis_with_mcs.mcs_labels
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert any("0.0-10.0" in label for label in labels)

        # Test properties columns
        df = analysis_with_mcs.get_mcs_properties()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.index.names == ["mcs_label", "organelle"]
        assert len(df.index) > 0
        expected_columns = [
            "n_contacts",
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
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Test values
        assert (df["n_contacts"] >= 0).all()
        assert (df["total_area"] >= 0).all()
        assert (df["min_dist"] >= 0).all()
        assert (df["max_dist"] >= 0).all()
        assert (df["std_area"] >= 0).all()
        assert (df["std_dist"] >= 0).all()

        # Test overview
        overview = analysis_with_mcs.get_mcs_overview()
        assert isinstance(overview, pd.DataFrame)
        assert not overview.empty
        assert isinstance(overview.index, pd.MultiIndex)
        groups = set(overview.index.get_level_values(0))
        assert groups.intersection({"overall", "per organelle", "per mcs"})
        for measure in [("per mcs", "mean_area"), ("per mcs", "mean_dist")]:
            if measure in overview.index:
                values = overview.loc[measure].dropna()
                assert not values.empty
                assert not np.isnan(values.iloc[0])

        df1 = analysis_with_mcs.get_dataframe()
        df2 = analysis_with_mcs.get_mcs_properties()
        pd.testing.assert_frame_equal(df1, df2)

    def test_get_mcs_properties_aggregates_match_patch_table(
        self, analysis_with_mcs
    ):
        """Aggregate counts (patches, partners) agree with the patch table."""
        props = analysis_with_mcs.get_mcs_properties()
        assert "n_partners" in props.columns
        assert "partners" in props.columns
        assert not props.index.duplicated().any()

        patches = analysis_with_mcs.get_mcs_patches()
        n_contacts = patches.groupby(["mcs_label", "organelle"]).size()
        n_partners = patches.groupby(["mcs_label", "organelle"])[
            "partner"
        ].nunique()

        for (label, organelle), row in props.iterrows():
            key = (label, organelle)
            assert row["n_contacts"] == n_contacts.loc[key]
            assert row["n_partners"] == n_partners.loc[key]
            assert len(row["partners"]) == row["n_partners"]


class TestMcsAnalysisFiltering:
    """Tests for Mcs_Analysis filtering behavior."""

    @pytest.fixture
    def analysis_with_multiple_mcs(self, project_with_sources):
        """Fixture providing an Mcs_Analysis with multiple MCS labels."""
        project_with_sources.search_mcs(max_distance=5.0, min_distance=0.0)
        project_with_sources.search_mcs(max_distance=10.0, min_distance=5.0)
        analysis = Mcs_Analysis(project_with_sources)
        return analysis

    def test_with_multiple_mcs(self, analysis_with_multiple_mcs):
        """Test that filtering by mcs_label reduces the number of results."""
        all_labels = analysis_with_multiple_mcs.mcs_labels
        assert len(all_labels) >= 2

        analysis_with_multiple_mcs.set_filters(mcs_labels=[all_labels[0]])
        overview = analysis_with_multiple_mcs.get_mcs_overview()
        assert len(overview) > 0
        df = analysis_with_multiple_mcs.get_mcs_properties()
        unique_labels = df.index.get_level_values("mcs_label").unique()
        assert len(unique_labels) == 1
        assert unique_labels[0] == all_labels[0]

        analysis_with_multiple_mcs.set_filters(mcs_labels=["nonexistent_label"])
        with pytest.raises(RuntimeError, match="No matching mcs labels found"):
            analysis_with_multiple_mcs.get_mcs_properties()
