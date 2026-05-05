from dataclasses import dataclass
from typing import Optional
import numpy as np
from pathlib import Path
from organelle_morphology.records import (
    PropertyBlock,
    Record,
)
import pytest


@dataclass(eq=False)
class MockProp(PropertyBlock):
    mock_str: str
    mock_int: int
    mock_path: Path
    mock_np: Optional[np.ndarray | Path]


@pytest.fixture
def mock_prop():
    return MockProp(
        mock_str="a", mock_int=5, mock_path=Path("../test_file.txt"), mock_np=None
    )


@pytest.fixture
def mock_prop_np():
    return MockProp(
        mock_str="a",
        mock_int=5,
        mock_path=Path("../test_file.txt"),
        mock_np=np.array([1.1, 5, 2, 99.5]),
    )


def test_stat_to_dict(mock_prop):
    stat = Record(data=mock_prop, meta=mock_prop)
    assert isinstance(stat.to_dict(), dict)


def test_stat_save_load(mock_prop, tmp_path):
    stat = Record(data=mock_prop, meta=mock_prop)
    file = tmp_path / "stat.yaml"
    assert not file.exists()
    stat.save_yaml(file)
    assert file.exists()

    loaded_stat = Record.from_yaml(file)
    assert loaded_stat.data == stat.data
    assert loaded_stat.meta == stat.meta


def test_stat_save_load_np(mock_prop, mock_prop_np, tmp_path):
    rec = Record(data=mock_prop_np, meta=mock_prop)
    file = tmp_path / "stat.yaml"
    assert not file.exists()
    rec.save_yaml(file)
    assert file.exists()

    loaded_rec = Record.from_yaml(file)
    assert loaded_rec.data == rec.data
    assert loaded_rec.meta == rec.meta


def test_registry_save_load_real_records(project_with_sources):
    """Test bulk saving and loading using actual production records."""
    project = project_with_sources

    project.calculate_profiles(method="Fixed Axis", ids="mito_0007", num_slices=3)

    original_records = project.registry.get_all().copy()
    assert len(original_records) > 0, "Pipeline failed to generate records."

    project.registry.save_all_to_yaml()
    assert (project.path / "analysis").exists()

    project.registry.clear()
    assert len(project.registry.get_all()) == 0

    project.registry.load_all_from_yaml()
    loaded_records = project.registry.get_all()
    assert len(loaded_records) == len(original_records)
    for orig, loaded in zip(original_records, loaded_records):
        assert orig == loaded
