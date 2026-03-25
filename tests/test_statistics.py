import pandas as pd
from organelle_morphology.statistics import Statistics


def test_statistics_defaults(project_with_sources):
    """Verify that we get a basic dataframe with default properties."""
    stats = Statistics(project_with_sources)
    df = stats.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ID" in df.columns
    # Default properties check
    assert "mesh_volume" in df.columns
    assert "sphericity" in df.columns
    assert df.columns[0] == "ID"


def test_statistics_alphabetical_sorting(project_with_sources):
    """Check that columns are sorted and ID filtering works."""
    stats = Statistics(project_with_sources)
    props = ["water_tight", "mesh_area", "mesh_volume"]  # non-alphabetical order
    df = stats.get_dataframe(ids="mito_0001", properties=props)
    assert df.shape[0] == 1
    assert df.iloc[0]["ID"] == "mito_0001"
    expected_order = [
        "ID",
        "mesh_area",
        "mesh_volume",
        "water_tight",
    ]  # alphabetical order
    assert list(df.columns) == expected_order


def test_statistics_summary_dataframe(project_with_sources):
    """Test the statistical summary logic, including boolean handling."""
    stats = Statistics(project_with_sources)
    # water_tight is boolean
    df_data = stats.get_dataframe(
        properties=["mesh_volume", "water_tight", "sphericity"]
    )
    summary_df = stats.get_summary_dataframe(df_data)
    assert "Measure" in summary_df.columns
    assert "Average (or Share)" in summary_df["Measure"].values
    avg_row = summary_df[summary_df["Measure"] == "Average (or Share)"].iloc[0]
    assert isinstance(avg_row["mesh_volume"], float)  # Volume should be a float
    assert 0.0 <= avg_row["water_tight"] <= 1.0  # Share of water_tight
    std_row = summary_df[summary_df["Measure"] == "Std Dev"].iloc[0]
    assert pd.isna(
        std_row["water_tight"]
    )  # Standard deviation should be NaN for boolean columns


def test_statistics_mcs_aggregation(project_with_sources, mocker):
    """Verify that MCS data is correctly pulled from the mcs_dict."""
    stats = Statistics(project_with_sources)
    org = project_with_sources.get_organelles("mito_0001")[0]  # organelle to mock
    # Instead of running MembraneContactSiteCalculator, mock the mcs_dict property
    test_mcs_data = {"0-0.01": {"total_area": 150.5, "mean_dist": 42.0}}
    mocker.patch.object(
        type(org),
        "mcs_dict",
        new_callable=mocker.PropertyMock,
        return_value=test_mcs_data,
    )
    props = ["total_area", "mean_dist", "mesh_volume"]  # Request contact keys
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
    stats = Statistics(project_with_sources)
    stats_df = stats.get_dataframe(properties=stats.get_properties())  # all properties
    assert not stats_df.empty
    for prop in stats.get_properties():
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
