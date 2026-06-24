import pandas as pd
import pytest

from organelle_morphology.curvature_analysis import (
    CurvatureAnalysis,
    CurvatureData,
    CurvatureMetaData,
)
from organelle_morphology.records import Record


def test_curvature_stats_data(project_with_sources):
    """Test that curvature stats contain valid data."""
    calculator = CurvatureAnalysis(project_with_sources)
    calculator.calculate_curvature_stats(ids="mito_0007")

    # Retrieve from the central registry
    for s in project_with_sources.records:
        if isinstance(s.data, CurvatureData) and s.meta.organelle_id == "mito_0007":
            stat = s
            break
    else:
        pytest.fail("Curvature stat for mito_0007 not found in project registry")

    assert isinstance(stat, Record)
    assert isinstance(stat.data, CurvatureData)
    assert isinstance(stat.meta, CurvatureMetaData)

    # Meta fields
    assert stat.meta.organelle_id == "mito_0007"
    assert stat.meta.curvature_radius > 0
    assert stat.meta.num_vertices > 0

    # Data fields should all be finite
    data = stat.data
    assert data.min_curvature <= data.max_curvature
    assert data.mean_curvature <= data.max_curvature
    assert data.mean_curvature >= data.min_curvature
    assert data.median_curvature <= data.max_curvature
    assert data.median_curvature >= data.min_curvature
    assert data.mean_absolute_curvature >= 0
    assert data.std_curvature >= 0


def test_curvature_stats_dataframe(project_with_sources):
    """Test that the CurvatureAnalysis generates the correct DataFrame."""
    calculator = CurvatureAnalysis(project_with_sources)
    calculator.calculate_curvature_stats(ids="mito_0007")

    df = calculator.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    assert df["ID"].iloc[0] == "mito_0007"

    expected_columns = {
        "ID",
        "min_curvature",
        "max_curvature",
        "mean_curvature",
        "std_curvature",
        "median_curvature",
        "mean_absolute_curvature",
        "num_vertices",
    }
    assert expected_columns.issubset(df.columns)


def test_curvature_stats_multiple_organelles(project_with_sources):
    """Test curvature stats across multiple organelles."""
    project_with_sources.set_curvature_radius(1)
    calculator = CurvatureAnalysis(project_with_sources)
    calculator.calculate_curvature_stats(ids="*")
    df = calculator.get_dataframe()

    project_with_sources.set_curvature_radius(3)
    calculator.calculate_curvature_stats(ids="mito_0007")
    df2 = calculator.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 19
    assert len(df2) == 20
    assert all(df2.min_curvature > -1)
    assert all(df2.min_curvature < 1)


def test_curvature_stats_no_match(project_with_sources):
    """Test that calculating stats with no matching organelles is handled gracefully."""
    calculator = CurvatureAnalysis(project_with_sources)
    calculator.calculate_curvature_stats(ids="nonexistent_*")

    df = calculator.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert len(df.columns) == 7


def test_curvature_stats_recompute(project_with_sources):
    """Test that recompute flag works (no duplicate records)."""
    calculator = CurvatureAnalysis(project_with_sources)
    calculator.calculate_curvature_stats(ids="mito_0007")

    count_before = sum(
        1
        for s in project_with_sources.records
        if isinstance(s.data, CurvatureData) and s.meta.organelle_id == "mito_0007"
    )

    # Recompute should not add duplicates
    calculator.calculate_curvature_stats(ids="mito_0007")

    count_after = sum(
        1
        for s in project_with_sources.records
        if isinstance(s.data, CurvatureData) and s.meta.organelle_id == "mito_0007"
    )

    assert count_after == count_before == 1, "Recompute created duplicate records"


def test_curvature_analysis_summary(project_with_sources):
    """Test that get_summary_dataframe works on curvature stats."""
    calculator = CurvatureAnalysis(project_with_sources)
    calculator.calculate_curvature_stats(ids="mito_0007")

    df = calculator.get_dataframe()
    summary = calculator.get_summary_dataframe(df)

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) > 0
    assert "Measure" in summary.columns or summary.index.name == "Measure"
