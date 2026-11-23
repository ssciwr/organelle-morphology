import os
import re
import subprocess
import sys
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


def test_get_hash_is_stable_hex(project_with_sources):
    """get_hash returns a 64-char sha256 hex string, stable within a process."""
    p = project_with_sources
    pc = ProfileCalculator(p)
    pc.calculate_profile_lengths(ids="mito_0007", axis="z", num_slices=3)

    rec = p.registry.get_all()[0]
    h = rec.get_hash()
    assert isinstance(h, str)
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    assert rec.get_hash() == h


# Builds the same logical identity (incl. non-empty blacklist/whitelist) and
# prints its filename digest. Run under different PYTHONHASHSEED values to prove
# the digest is process-independent (the core of issue #7).
_HASH_SCRIPT = """
from dataclasses import dataclass
from pathlib import Path

from organelle_morphology.project import ProjectMetadata
from organelle_morphology.records import PropertyBlock, Record
from organelle_morphology.source import SourceMetadata


@dataclass(eq=False, unsafe_hash=True)
class MockProp(PropertyBlock):
    mock_str: str
    mock_int: int
    mock_path: Path
    mock_np: object


meta = MockProp("a", 5, Path("../test_file.txt"), None)
meta_project = ProjectMetadata(
    path=Path("/tmp/proj"),
    name="proj",
    clipping=None,
    compression="s0",
    sources=("synth_data",),
    blacklist=("a", "m", "z"),
    whitelist=("b", "y"),
)
meta_sources = (
    SourceMetadata(
        data_root=Path("/tmp/data"),
        downsampling=((1, 1, 1),),
        levels=("s0",),
        size=(10, 10, 10),
        resolution=(0.01, 0.01, 0.01),
        unit="um",
        name="synth_data",
        coarse_level="s0",
    ),
)
rec = Record(
    data=meta, meta=meta, project=None,
    meta_sources=meta_sources, meta_project=meta_project,
)
print(rec.get_hash())
"""


def _hash_in_subprocess(seed: int) -> str:
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _HASH_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_get_hash_stable_across_processes():
    """Same identity -> same filename across PYTHONHASHSEED (issue #7)."""
    assert _hash_in_subprocess(1) == _hash_in_subprocess(2)


def test_project_metadata_sorts_filters(project):
    """blacklist/whitelist are sorted at construction (set-order is salted)."""
    project.permanent_blacklist = {"z", "a", "m"}
    project.permanent_whitelist = {"y", "b"}
    md = project.metadata
    assert md.blacklist == ("a", "m", "z")
    assert md.whitelist == ("b", "y")


def test_get_hash_changes_with_meta(project_with_sources):
    """Different metadata -> different filename."""
    p = project_with_sources
    pc = ProfileCalculator(p)
    pc.calculate_profile_lengths(ids="mito_0007", axis="z", num_slices=3)
    pc.calculate_profile_lengths(ids="mito_0008", axis="z", num_slices=3)

    recs = p.registry.get_all()
    assert recs[0].get_hash() != recs[1].get_hash()


def test_resave_same_identity_overwrites(project_with_sources):
    """Same identity (even with changed data) -> same filename, updated content."""
    p = project_with_sources
    pc = ProfileCalculator(p)
    pc.calculate_profile_lengths(ids="mito_0007", axis="z", num_slices=3)
    rec = p.registry.get_all()[0]
    d = p.path / "analysis" / rec.name

    p.registry.save_all_to_yaml()
    files = list(d.glob("*.yaml"))
    assert len(files) == 1
    assert files[0].stem == rec.get_hash()
    first = files[0].read_text()

    # Change data but keep identity: data is excluded from the digest.
    rec.data.mean_width = rec.data.mean_width + 1.0
    p.registry.save_all_to_yaml()
    files = list(d.glob("*.yaml"))
    assert len(files) == 1
    assert files[0].stem == rec.get_hash()
    assert files[0].read_text() != first


def test_loader_ignores_legacy_filenames(project_with_sources):
    """Loader loads only stable-hex stems, ignoring salted-int / UUID files."""
    p = project_with_sources
    pc = ProfileCalculator(p)
    pc.calculate_profile_lengths(ids="mito_0007", axis="z", num_slices=3)
    pc.calculate_profile_lengths(ids="mito_0008", axis="z", num_slices=3)
    rec_new, rec_old = p.registry.get_all()

    d = p.path / "analysis" / rec_new.name
    d.mkdir(parents=True, exist_ok=True)
    rec_new.save_yaml(d / f"{rec_new.get_hash()}.yaml")  # stable hex
    rec_old.save_yaml(d / "1234567890123456789.yaml")  # legacy salted-int
    rec_old.save_yaml(d / "16f45908-3887-4668-b916-e987f8b3e145.yaml")  # legacy UUID

    p.registry.clear()
    p.registry.load_all_from_yaml()

    loaded = p.registry.get_all()
    assert len(loaded) == 1
    assert loaded[0].meta.organelle_id == "mito_0007"
