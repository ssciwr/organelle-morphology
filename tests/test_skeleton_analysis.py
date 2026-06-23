from dataclasses import fields
from typing import Optional

import pandas as pd

from organelle_morphology.records import Record
from organelle_morphology.skeleton_analysis import (
    SkeletonMetaData,
    SkeletonData,
    Skeleton_Analysis,
)


def _make_skeleton_record(
    project,
    organelle_id: str = "mito_0001",
    method: str = "wavefront",
    theta: float = 45.0,
    waves: Optional[int] = None,
    step_size: Optional[int] = None,
    epsilon: Optional[float] = None,
    sampling_dist: Optional[float] = None,
    **data_kwargs,
) -> Record:
    """Helper to create a SkeletonData record with default values."""
    meta = SkeletonMetaData(
        organelle_id=organelle_id,
        method=method,
        theta=theta,
        path_sample_dist=0.5,
        waves=waves,
        step_size=step_size,
        epsilon=epsilon,
        sampling_dist=sampling_dist,
    )
    data_defaults = {
        "num_nodes": 100,
        "total_length": 50.0,
        "std_length": 2.5,
        "num_branch_points": 10,
        "end_points": 5,
        "mean_length": 0.5,
        "longest_path": 12.0,
        "mean_radius": 1.2,
        "std_radius": 0.3,
    }
    data_defaults.update(data_kwargs)
    data = SkeletonData(**data_defaults)
    return Record(data=data, meta=meta, project=project)


def test_get_dataframe_empty(project_with_sources):
    """Verify that get_dataframe returns an empty DataFrame with correct columns when no records exist."""
    stats = Skeleton_Analysis(project_with_sources)
    df = stats.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected_cols = (
        ["organelle_id"]
        + [f.name for f in fields(SkeletonData)]
        + [f.name for f in fields(SkeletonMetaData) if f.name != "organelle_id"]
    )
    len(expected_cols)
    assert list(df.columns) == expected_cols


def test_get_summary_dataframe(project_with_sources):
    """Verify that get_summary_dataframe computes correct statistics."""
    for i in range(5):
        record = _make_skeleton_record(
            project_with_sources,
            organelle_id=f"mito_{i:04d}",
            num_nodes=100 + i * 10,
            total_length=50.0 + i * 10.0,
        )
        project_with_sources.registry.add(record)

    stats = Skeleton_Analysis(project_with_sources)
    summary = stats.get_summary_dataframe()

    assert not summary.empty
    assert "Measure" in summary.columns
    assert "num_nodes" in summary.columns
    assert "total_length" in summary.columns
    # Verify the measure names are present in the Measure column
    measures = summary["Measure"].tolist()
    assert "Average (or Share)" in measures
    assert "Std Dev" in measures
    assert "Median (50th percentile)" in measures
    assert "Minimum" in measures
    assert "Maximum" in measures


def test_get_summary_dataframe_with_filtered_df(project_with_sources):
    """Verify that get_summary_dataframe works with a custom filtered DataFrame."""
    for i in range(5):
        record = _make_skeleton_record(
            project_with_sources,
            organelle_id=f"mito_{i:04d}",
            num_nodes=100 + i * 10,
            total_length=50.0 + i * 10.0,
        )
        project_with_sources.registry.add(record)

    stats = Skeleton_Analysis(project_with_sources)
    df = stats.get_dataframe()

    filtered_df = df[df["num_nodes"] > 110]
    # Filter to only 2 records
    summary = stats.get_summary_dataframe(df=filtered_df)

    assert not summary.empty
    # The summary should reflect only the filtered data
    avg_row = summary[summary["Measure"] == "Average (or Share)"]
    assert avg_row["num_nodes"].iloc[0] > 110


def test_get_summary_dataframe_empty(project_with_sources):
    """Verify that get_summary_dataframe returns empty DataFrame when no records exist."""
    stats = Skeleton_Analysis(project_with_sources)
    summary = stats.get_summary_dataframe()
    assert summary.empty


def test_get_summary_dataframe_from_dataframe_empty(project_with_sources):
    """Verify that get_summary_dataframe returns empty when passed an empty DataFrame."""
    stats = Skeleton_Analysis(project_with_sources)
    empty_df = pd.DataFrame()
    summary = stats.get_summary_dataframe(df=empty_df)
    assert summary.empty


def test_save_records(project_with_sources, tmp_path):
    """Verify that save_records writes records to YAML."""
    record = _make_skeleton_record(project_with_sources, organelle_id="mito_0001")
    project_with_sources.registry.add(record)

    stats = Skeleton_Analysis(project_with_sources)
    stats.save_records()

    # The project's tmp_path should contain a YAML file with the record
    yaml_files = list(tmp_path.glob("**/*.yaml"))
    assert len(yaml_files) > 0


def test_get_dataframe_mixed_optional_fields(project_with_sources):
    """Verify correct column handling when records have different subsets of optional fields."""
    record1 = _make_skeleton_record(
        project_with_sources,
        organelle_id="mito_0001",
        waves=10,
        step_size=5,
    )
    record2 = _make_skeleton_record(
        project_with_sources,
        organelle_id="mito_0002",
        epsilon=0.01,
        sampling_dist=1.0,
    )
    project_with_sources.registry.add(record1)
    project_with_sources.registry.add(record2)

    stats = Skeleton_Analysis(project_with_sources)
    df = stats.get_dataframe()

    assert df.shape[0] == 2
    # All optional columns should be present
    assert "waves" in df.columns
    assert "step_size" in df.columns
    assert "epsilon" in df.columns
    assert "sampling_dist" in df.columns
    # Record 1 should have NaN for epsilon and sampling_dist
    assert pd.isna(df[df["organelle_id"] == "mito_0001"]["epsilon"].iloc[0])
    assert pd.isna(df[df["organelle_id"] == "mito_0001"]["sampling_dist"].iloc[0])
    # Record 2 should have NaN for waves and step_size
    assert pd.isna(df[df["organelle_id"] == "mito_0002"]["waves"].iloc[0])
    assert pd.isna(df[df["organelle_id"] == "mito_0002"]["step_size"].iloc[0])


def test_get_dataframe_with_records(project_with_sources):
    """Verify that get_dataframe correctly populates columns from records."""
    record = _make_skeleton_record(project_with_sources, organelle_id="mito_0001")
    project_with_sources.registry.add(record)

    stats = Skeleton_Analysis(project_with_sources)
    df = stats.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == 1
    assert df.iloc[0]["organelle_id"] == "mito_0001"

    # Check skeleton data columns
    assert df.iloc[0]["num_nodes"] == 100
    assert df.iloc[0]["total_length"] == 50.0
    assert df.iloc[0]["std_length"] == 2.5
    assert df.iloc[0]["num_branch_points"] == 10
    assert df.iloc[0]["end_points"] == 5
    assert df.iloc[0]["mean_length"] == 0.5
    assert df.iloc[0]["longest_path"] == 12.0
    assert df.iloc[0]["mean_radius"] == 1.2
    assert df.iloc[0]["std_radius"] == 0.3

    # Check metadata columns
    assert df.iloc[0]["method"] == "wavefront"
    assert df.iloc[0]["theta"] == 45.0
    assert df.iloc[0]["path_sample_dist"] == 0.5
    assert pd.isna(df.iloc[0]["waves"])
    assert pd.isna(df.iloc[0]["step_size"])
    assert pd.isna(df.iloc[0]["epsilon"])
    assert pd.isna(df.iloc[0]["sampling_dist"])


def test_get_dataframe_with_optional_fields(project_with_sources):
    """Verify that optional metadata fields appear in the DataFrame when provided."""
    record = _make_skeleton_record(
        project_with_sources,
        organelle_id="mito_0002",
        method="vertex_cluster",
        waves=10,
        step_size=5,
        epsilon=0.01,
        sampling_dist=1.0,
    )
    project_with_sources.registry.add(record)

    stats = Skeleton_Analysis(project_with_sources)
    df = stats.get_dataframe()

    assert not df.empty
    assert df.iloc[0]["waves"] == 10
    assert df.iloc[0]["step_size"] == 5
    assert df.iloc[0]["epsilon"] == 0.01
    assert df.iloc[0]["sampling_dist"] == 1.0


def test_get_dataframe_multiple_records(project_with_sources):
    """Verify that multiple records are correctly aggregated into the DataFrame."""
    record1 = _make_skeleton_record(project_with_sources, organelle_id="mito_0001")
    record2 = _make_skeleton_record(
        project_with_sources,
        organelle_id="mito_0002",
        method="vertex_cluster",
        theta=90.0,
        waves=20,
        step_size=10,
    )
    project_with_sources.registry.add(record1)
    project_with_sources.registry.add(record2)

    stats = Skeleton_Analysis(project_with_sources)
    df = stats.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == 2
    assert set(df["organelle_id"]) == {"mito_0001", "mito_0002"}

    mito_0001 = df[df["organelle_id"] == "mito_0001"].iloc[0]
    assert mito_0001["method"] == "wavefront"
    assert mito_0001["theta"] == 45.0

    mito_0002 = df[df["organelle_id"] == "mito_0002"].iloc[0]
    assert mito_0002["method"] == "vertex_cluster"
    assert mito_0002["theta"] == 90.0
    assert mito_0002["waves"] == 20
    assert mito_0002["step_size"] == 10
