from dataclasses import dataclass
import numpy as np
from pathlib import Path
import yaml
from organelle_morphology.statistics import (
    PropertyBlock,
    Stats,
)
import pytest


@dataclass
class MockProp(PropertyBlock):
    mock_str: str
    mock_int: int
    mock_path: Path
    mock_np: np.ndarray


yaml.add_constructor(
    "tag:yaml.org,2002:python/object/apply:tests.test_statistics.MockProp",
    lambda loader, node: PropertyBlock.yaml_constructor(loader, node, MockProp),
    Loader=yaml.SafeLoader,
)


@pytest.fixture
def mock_prop():
    return MockProp("a", 5, Path("../test_file.txt"), [1, 2.2, 3])


def test_stat_to_dict(mock_prop):
    stat = Stats(data=mock_prop, meta=mock_prop)
    assert isinstance(stat.to_dict(), dict)


def test_stat_save_load(mock_prop, tmp_path):
    stat = Stats(data=mock_prop, meta=mock_prop)
    file = tmp_path / "stat.yaml"
    assert not file.exists()
    stat.save_yaml(file)
    assert file.exists()

    loaded_stat = Stats.from_yaml(file)
    assert loaded_stat.data == stat.data
    assert loaded_stat.meta == stat.meta
