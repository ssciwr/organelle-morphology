import pandas as pd

from organelle_morphology.analysis import Misc_Analysis
from organelle_morphology.statistics import PropertyBlock, Record
from organelle_morphology.organelle import McsProperties, McsMeta


def test_statistics_defaults(project_with_sources):
    """Verify that we get a basic dataframe with default properties."""
    stats = Misc_Analysis(project_with_sources, PropertyBlock)
    df = stats.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ID" in df.columns
    # Default properties check
    assert "volume" in df.columns
    assert "sphericity" in df.columns
    assert df.columns[0] == "ID"


def test_statistics_alphabetical_sorting(project_with_sources):
    """Check that columns are sorted and ID filtering works."""
    stats = Misc_Analysis(project_with_sources, PropertyBlock)
    props = ["water_tight", "area", "volume"]  # non-alphabetical order
    df = stats.get_dataframe(ids="mito_0001", properties=props)
    assert df.shape[0] == 1
    assert df.iloc[0]["ID"] == "mito_0001"
    expected_order = [
        "ID",
        "area",
        "volume",
        "water_tight",
    ]  # alphabetical order
    assert list(df.columns) == expected_order


def test_statistics_summary_dataframe(project_with_sources):
    """Test the statistical summary logic, including boolean handling."""
    stats = Misc_Analysis(project_with_sources, PropertyBlock)
    # water_tight is boolean
    df_data = stats.get_dataframe(properties=["volume", "water_tight", "sphericity"])
    summary_df = stats.get_summary_dataframe(df_data)
    assert "Measure" in summary_df.columns
    assert "Average (or Share)" in summary_df["Measure"].values
    avg_row = summary_df[summary_df["Measure"] == "Average (or Share)"].iloc[0]
    assert isinstance(avg_row["volume"], float)  # Volume should be a float
    assert 0.0 <= avg_row["water_tight"] <= 1.0  # Share of water_tight
    std_row = summary_df[summary_df["Measure"] == "Std Dev"].iloc[0]
    assert pd.isna(
        std_row["water_tight"]
    )  # Standard deviation should be NaN for boolean columns


def test_statistics_mcs_aggregation(project_with_sources):
    """Verify that MCS data is correctly pulled from the mcs_dict."""

    # Create real dataclasses to pass the isinstance(stat.data, PropertyBlock) check natively
    meta = McsMeta(mcs_label="0-0.01", organelle_id="mito_0001", max_dist=0.01)
    data = McsProperties(
        n_contacts=1,
        total_area=150.5,
        mean_area=150.5,
        std_area=0.0,
        mean_dist=42.0,
        std_dist=0.0,
        n_contacts_per_area=0.1,
        n_contacts_per_volume=0.1,
        area_per_area=0.1,
        area_per_volume=0.1,
    )
    stat = Record(data=data, meta=meta)

    # Inject the stat into the project BEFORE initializing Misc_Analysis so own_stats picks it up
    project_with_sources.add_stat(stat)

    stats = Misc_Analysis(project_with_sources, PropertyBlock)

    props = ["total_area", "mean_dist", "volume"]  # Request contact keys
    df = stats.get_dataframe(ids="mito_0001", properties=props)

    assert df.iloc[0]["0-0.01-total_area"] == 150.5  # Verify
    assert df.iloc[0]["0-0.01-mean_dist"] == 42.0  # Verify

    summary_df = stats.get_summary_dataframe(df)  # Verify the summary
    avg_row = summary_df[summary_df["Measure"] == "Average (or Share)"].iloc[0]
    assert avg_row["0-0.01-total_area"] == 150.5


def test_statistics_integration(project_with_sources):
    """Integration test for the full statistics pipeline."""
    # assume project_with_sources contains meshes
    project_with_sources.skeletonize_wavefront()
    project_with_sources.search_mcs(10)  # calc and add contact sites
    _ = project_with_sources.geometric_properties  # calc and add voxel based properties

    stats = Misc_Analysis(project_with_sources, PropertyBlock)

    # Combine all properties since get_properties() no longer exists
    all_props = (
        stats.get_mesh_properties()
        + stats.get_skeleton_properties()
        + stats.get_geometry_properties()
        + stats.get_contact_properties()
    )

    stats_df = stats.get_dataframe(properties=all_props)  # all properties
    assert not stats_df.empty

    for prop in all_props:
        if prop in stats.get_contact_properties():  # if dynamic property
            # generic property (e.g., "n_contacts") to dynamic column (e.g., "0-0.01-n_contacts")
            dynamic_cols = [col for col in stats_df.columns if prop in col]
            assert len(dynamic_cols) > 0, f"Column for {prop} was not generated."
            assert stats_df[dynamic_cols[0]].notna().any(), (
                f"Data for {prop} is entirely NaN."
            )
        else:  # if static property
            assert prop in stats_df.columns, (
                f"Static property {prop} is missing from DataFrame."
            )
            assert stats_df[prop].notna().all(), f"Some data for {prop} is NaN."
