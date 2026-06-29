from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from organelle_morphology.profile_calculations import ProfileCalculator
from organelle_morphology.records import PropertyBlock, Record


@dataclass(eq=False, unsafe_hash=True)
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


def test_stat_to_dict(mock_prop, project_with_sources):
    stat = Record(data=mock_prop, meta=mock_prop, project=project_with_sources)
    assert isinstance(stat.to_dict(), dict)


def test_stat_save_load(mock_prop, tmp_path, project_with_sources):
    stat = Record(data=mock_prop, meta=mock_prop, project=project_with_sources)
    file = tmp_path / "stat.yaml"
    assert not file.exists()
    stat.save_yaml(file)
    assert file.exists()

    loaded_stat = Record.from_yaml(file)
    assert loaded_stat.data == stat.data
    assert loaded_stat.meta == stat.meta


def test_stat_save_load_np(mock_prop, mock_prop_np, tmp_path, project_with_sources):
    rec = Record(data=mock_prop_np, meta=mock_prop, project=project_with_sources)
    file = tmp_path / "stat.yaml"
    assert not file.exists()
    rec.save_yaml(file)
    assert file.exists()

    loaded_rec = Record.from_yaml(file)
    assert loaded_rec.data == rec.data
    assert loaded_rec.meta == rec.meta


def test_registry_save_load_real_records(project_with_sources):
    """Test bulk saving and loading using actual production records."""
    p = project_with_sources

    pc = ProfileCalculator(p)
    pc.calculate_profile_lengths(ids="mito_0007", axis="z", num_slices=3)

    original_records = p.registry.get_all().copy()
    assert len(original_records) > 0, "Pipeline failed to generate records."

    p.registry.save_all_to_yaml()
    assert (p.path / "analysis").exists()
    assert len(list((p.path / "analysis").iterdir())) == 1

    # test overwriting
    p.registry.save_all_to_yaml()
    assert len(list((p.path / "analysis").iterdir())) == 1

    p.registry.clear()
    assert len(p.registry.get_all()) == 0

    p.registry.load_all_from_yaml()
    loaded_records = p.registry.get_all()
    assert len(loaded_records) == len(original_records)
    for orig, loaded in zip(original_records, loaded_records):
        assert orig == loaded


def test_registry_clear_record(project_with_sources):
    """Test removing a specific record from the registry."""
    p = project_with_sources

    pc = ProfileCalculator(p)
    pc.calculate_profile_lengths(ids="mito_*", axis="z", num_slices=3)

    original_records = p.registry.get_all()
    n_original = len(original_records)
    assert n_original == 19

    record_to_remove = original_records[0]

    # Verify it's indexed correctly before removal
    record_type = type(record_to_remove.data).__name__
    assert record_to_remove in p.registry.get_by_type(record_type)

    p.registry.clear_record(record_to_remove)

    # Verify removal
    assert record_to_remove not in p.registry.get_all()
    assert record_to_remove not in p.registry.get_by_type(record_type)

    # Verify other records remain
    assert len(p.registry.get_all()) == n_original - 1


def test_registry_clear_record_not_found(project_with_sources):
    """Test that clear_record warns gracefully when record isn't in registry."""
    p = project_with_sources

    pc = ProfileCalculator(p)
    pc.calculate_profile_lengths(ids="mito_0007", axis="z", num_slices=3)

    original_count = len(p.registry.get_all())

    # Create a fresh record that was never added
    mock_prop = MockProp(
        mock_str="a", mock_int=5, mock_path=Path("../test_file.txt"), mock_np=None
    )
    orphan_record = Record(data=mock_prop, meta=mock_prop, project=p)

    p.registry.clear_record(orphan_record)

    # Count should be unchanged
    assert len(p.registry.get_all()) == original_count
