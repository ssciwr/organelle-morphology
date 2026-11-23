import pandas as pd

from organelle_morphology.analysis import (
    STATIC_COLUMN_GROUP,
    Misc_Analysis,
)
from organelle_morphology.records import Record, PropertyBlock
from organelle_morphology.organelle import (
    CURRENT_MCS_SCHEMA_VERSION,
    McsData,
    McsMetadata,
)


def test_statistics_defaults(project_with_sources):
    """Verify that we get a basic dataframe with default properties."""
    stats = Misc_Analysis(project_with_sources, PropertyBlock)
    df = stats.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["group", "property"]
    assert (STATIC_COLUMN_GROUP, "ID") in df.columns
    # Default properties check
    assert (STATIC_COLUMN_GROUP, "volume") in df.columns
    assert (STATIC_COLUMN_GROUP, "sphericity") in df.columns
    assert df.columns[0] == (STATIC_COLUMN_GROUP, "ID")


def test_statistics_alphabetical_sorting(project_with_sources):
    """Check that columns are sorted and ID filtering works."""
    stats = Misc_Analysis(project_with_sources, PropertyBlock)
    props = ["water_tight", "area", "volume"]  # non-alphabetical order
    df = stats.get_dataframe(ids="mito_0001", properties=props)
    assert df.shape[0] == 1
    assert df.iloc[0][(STATIC_COLUMN_GROUP, "ID")] == "mito_0001"
    expected_order = [
        (STATIC_COLUMN_GROUP, "ID"),
        (STATIC_COLUMN_GROUP, "area"),
        (STATIC_COLUMN_GROUP, "volume"),
        (STATIC_COLUMN_GROUP, "water_tight"),
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
    assert isinstance(avg_row[(STATIC_COLUMN_GROUP, "volume")], float)
    assert 0.0 <= avg_row[(STATIC_COLUMN_GROUP, "water_tight")] <= 1.0
    std_row = summary_df[summary_df["Measure"] == "Std Dev"].iloc[0]
    assert pd.isna(
        std_row[(STATIC_COLUMN_GROUP, "water_tight")]
    )  # Standard deviation should be NaN for boolean columns


def test_statistics_mcs_aggregation(project_with_sources):
    """Verify that MCS data is pulled from the mcs_dict with structured columns."""
    label = "0-0.1,mito-er"

    # Create real dataclasses to pass the isinstance(stat.data, PropertyBlock) check natively
    meta = McsMetadata(
        mcs_label=label,
        organelle_id="mito_0001",
        max_dist=0.1,
        schema_version=CURRENT_MCS_SCHEMA_VERSION,
    )
    data = McsData(
        n_contacts=1,
        n_partners=2,
        total_area=150.5,
        mean_area=150.5,
        std_area=0.0,
        mean_dist=42.0,
        std_dist=0.0,
        min_dist=0.1,
        max_dist=0.2,
        partners=("other_one", "other_two"),
        min_dist_per_org=(0.1, 0.2),
        max_dist_per_org=(0.1, 0.2),
        n_contacts_per_area=0.1,
        n_contacts_per_volume=0.1,
        area_per_area=0.1,
        area_per_volume=0.1,
    )
    stat = Record(data=data, meta=meta, project=project_with_sources)

    # Inject the stat into the project BEFORE initializing Misc_Analysis so own_records picks it up
    project_with_sources.registry.add(stat)

    stats = Misc_Analysis(project_with_sources, PropertyBlock)

    contact_props = set(stats.get_contact_properties())
    assert {"n_partners", "min_dist", "max_dist"} <= contact_props

    props = ["total_area", "mean_dist", "n_partners", "volume"]  # Request contact keys
    df = stats.get_dataframe(ids="mito_0001", properties=props)

    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["group", "property"]
    assert ("static", "ID") in df.columns
    assert (label, "total_area") in df.columns
    assert (label, "mean_dist") in df.columns
    assert (label, "n_partners") in df.columns
    assert df.iloc[0][("static", "ID")] == "mito_0001"
    assert df.iloc[0][(label, "total_area")] == 150.5
    assert df.iloc[0][(label, "mean_dist")] == 42.0
    assert df.iloc[0][(label, "n_partners")] == 2

    # get_contact_stats exposes structured (mcs_label, property) keys, so a
    # hyphenated label such as "0-0.1,mito-er" never has to be parsed back out.
    organelle = project_with_sources.get_organelles("mito_0001")[0]
    raw = stats.get_contact_stats(
        organelle, {"total_area", "n_contacts", "n_partners"}
    )
    assert raw[(label, "total_area")] == 150.5
    assert raw[(label, "n_contacts")] == 1
    assert raw[(label, "n_partners")] == 2

    summary_df = stats.get_summary_dataframe(df)
    assert "Measure" in summary_df.columns
    avg_row = summary_df[summary_df["Measure"] == "Average (or Share)"].iloc[0]
    assert avg_row[(label, "total_area")] == 150.5


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
            # Structured columns: (mcs_label, property); the label is opaque.
            dynamic_cols = [
                col
                for col in stats_df.columns
                if isinstance(col, tuple)
                and col[0] != STATIC_COLUMN_GROUP
                and col[1] == prop
            ]
            assert len(dynamic_cols) > 0, f"Column for {prop} was not generated."

            # Volume-dependent contact metrics will be NaN if the contacting meshes are open
            if "volume" in prop:
                continue

            assert stats_df[dynamic_cols[0]].notna().any(), (
                f"Data for {prop} is entirely NaN."
            )
        else:  # if static property
            static_col = (STATIC_COLUMN_GROUP, prop)
            assert static_col in stats_df.columns, (
                f"Static property {prop} is missing from DataFrame."
            )

            # The mesh volume will legitimately be NaN for open meshes.
            if prop == "volume":
                # Strictly enforce the relationship between water_tight and NaN
                watertight_col = (STATIC_COLUMN_GROUP, "water_tight")
                if watertight_col in stats_df.columns:
                    watertight_mask = stats_df[watertight_col].eq(True)
                    open_mask = stats_df[watertight_col].eq(False)

                    if watertight_mask.any():
                        assert stats_df.loc[watertight_mask, static_col].notna().all(), (
                            f"{prop} is NaN even for completely watertight meshes."
                        )

                    if open_mask.any():
                        assert stats_df.loc[open_mask, static_col].isna().all(), (
                            f"{prop} is not NaN for open meshes."
                        )
            else:
                assert stats_df[static_col].notna().all(), (
                    f"Some data for {prop} is NaN."
                )
