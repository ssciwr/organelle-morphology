from dataclasses import dataclass
from typing import Optional
import numpy as np
from pathlib import Path
import yaml
from organelle_morphology.statistics import (
    Properties,
    Stats,
)
import pytest


@dataclass(eq=False)
class MockProp(Properties):
    mock_str: str
    mock_int: int
    mock_path: Path
    mock_np: Optional[np.ndarray | Path]


yaml.add_constructor(
    "tag:yaml.org,2002:python/object/apply:tests.test_statistics.MockProp",
    lambda loader, node: Properties.yaml_constructor(loader, node, MockProp),
    Loader=yaml.SafeLoader,
)


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


def test_stat_save_load_np(mock_prop, mock_prop_np, tmp_path):
    stat = Stats(data=mock_prop_np, meta=mock_prop)
    file = tmp_path / "stat.yaml"
    assert not file.exists()
    stat.save_yaml(file)
    assert file.exists()

    loaded_stat = Stats.from_yaml(file)
    assert loaded_stat.data == stat.data
    assert loaded_stat.meta == stat.meta
