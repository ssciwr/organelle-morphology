import pytest
import pandas as pd
import numpy as np

from organelle_morphology.mcs_analysis import Mcs_Analysis


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
