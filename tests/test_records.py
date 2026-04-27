from dataclasses import dataclass
import numpy as np
from pathlib import Path
import yaml
from organelle_morphology.records import (
    PropertyBlock,
    Record,
)
import pytest


@dataclass
class MockProp(PropertyBlock):
    mock_str: str
    mock_int: int
    mock_path: Path
    mock_np: np.ndarray


yaml.add_constructor(
    "tag:yaml.org,2002:python/object/apply:tests.test_records.MockProp",
    lambda loader, node: PropertyBlock.yaml_constructor(loader, node, MockProp),
    Loader=yaml.SafeLoader,
)


@pytest.fixture
def mock_prop():
    return MockProp("a", 5, Path("../test_file.txt"), [1, 2.2, 3])


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


def test_registry_save_load_real_records(project_with_sources, tmp_path):
    """Test bulk saving and loading using actual production records."""
    project = project_with_sources

    # Generate some profiles to populate the registry with ProfileData & ProfileMetadata
    project.calculate_profiles(method="Fixed Axis", ids="mito_0007", num_slices=3)

    # Keep copy of original records for comparison
    original_records = project.registry.get_all().copy()
    assert len(original_records) > 0, "Pipeline failed to generate records."

    # Save to the pytests temp path
    file_path = tmp_path / "real_records.yaml"
    project.registry.save_all_to_yaml(file_path)
    assert file_path.exists()

    # Clear the registry
    project.registry.clear()
    assert len(project.registry.get_all()) == 0

    # Load the records back and check them
    project.registry.load_all_from_yaml(file_path)
    loaded_records = project.registry.get_all()
    assert len(loaded_records) == len(original_records)
    for orig, loaded in zip(original_records, loaded_records):
        assert orig == loaded
