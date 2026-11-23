import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from organelle_morphology.organelle import (
    CURRENT_MCS_SCHEMA_VERSION,
    McsData,
    McsMetadata,
)
from organelle_morphology.profile_calculations import ProfileCalculator
from organelle_morphology.records import PropertyBlock, Record


@dataclass(eq=False, unsafe_hash=True)
class MockProp(PropertyBlock):
    mock_str: str
    mock_int: int
    mock_path: Path
    mock_np: Optional[np.ndarray | Path]


def _mcs_data():
    return McsData(
        n_contacts=2,
        n_partners=1,
        total_area=4.0,
        mean_area=2.0,
        std_area=1.0,
        mean_dist=0.05,
        std_dist=0.01,
        min_dist=0.02,
        max_dist=0.1,
        partners=("mito_0002",),
        min_dist_per_org=(0.02,),
        max_dist_per_org=(0.1,),
        n_contacts_per_area=0.1,
        n_contacts_per_volume=0.2,
        area_per_area=0.4,
        area_per_volume=0.8,
    )


def _mcs_meta(**overrides):
    kwargs = {
        "mcs_label": "0.0-0.1,-",
        "max_dist": 0.1,
        "min_dist": 0.0,
        "schema_version": CURRENT_MCS_SCHEMA_VERSION,
        "filter_1": "mito*",
        "filter_2": "er*",
        "organelle_id": "mito_0001",
    }
    kwargs.update(overrides)
    return McsMetadata(**kwargs)


def _write_record_without_schema_version(record, filename):
    """Save a record, then strip ``schema_version`` to mimic pre-fix YAML."""
    record.save_yaml(filename)
    text = "\n".join(
        line
        for line in filename.read_text().splitlines()
        if "schema_version:" not in line
    )
    filename.write_text(text + "\n")
    return filename


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


def test_mcs_record_schema_roundtrip(project_with_sources, tmp_path):
    """A current-version MCS record round-trips with tuple field types."""
    rec = Record(
        data=_mcs_data(), meta=_mcs_meta(), project=project_with_sources
    )
    file = tmp_path / "mcs.yaml"
    rec.save_yaml(file)

    loaded = Record.from_yaml(file)
    assert loaded.meta.schema_version == CURRENT_MCS_SCHEMA_VERSION
    assert loaded.meta == rec.meta
    assert loaded.data == rec.data
    assert isinstance(loaded.data.partners, tuple)
    assert isinstance(loaded.data.min_dist_per_org, tuple)
    assert isinstance(loaded.data.max_dist_per_org, tuple)


def test_mcs_metadata_rejects_old_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        _mcs_meta(schema_version=1)


def test_load_all_rejects_old_mcs_yaml_and_lists_files(
    project_with_sources, tmp_path, caplog
):
    p = project_with_sources
    rec = Record(data=_mcs_data(), meta=_mcs_meta(), project=p)

    record_dir = p.path / "analysis" / "McsData"
    valid_file = record_dir / "current.yaml"
    rec.save_yaml(valid_file)

    old_file = _write_record_without_schema_version(
        rec, record_dir / "without_version.yaml"
    )
    old_version_file = record_dir / "old_version.yaml"
    old_version_file.write_text(
        valid_file.read_text().replace(
            f"schema_version: {CURRENT_MCS_SCHEMA_VERSION}",
            "schema_version: 1",
        )
    )

    p.registry.clear()
    with caplog.at_level(logging.WARNING):
        p.registry.load_all_from_yaml()

    assert len(p.registry.get_all()) == 1
    assert p.registry.get_all()[0].meta == rec.meta
    assert old_file.name in caplog.text
    assert old_version_file.name in caplog.text


