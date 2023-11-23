from organelle_morphology.project import *
from organelle_morphology.organelle import Organelle

import pytest
import pathlib


def test_synthetic_data_generation(
    synthetic_data, cebra_project_path, cebra_project_original_meshes
):
    """Check that the synthetic data is correctly generated"""
    project_path, original_meshes = synthetic_data
    assert project_path
    assert original_meshes

    assert cebra_project_path == project_path
    assert cebra_project_original_meshes == original_meshes

    project_path = pathlib.Path(project_path)

    assert (project_path / "dataset.json").exists()
    assert (project_path / "images" / "bdv-n5" / "synth_data.n5").exists()
    assert (project_path / "images" / "bdv-n5" / "synth_data.xml").exists()

    organell_path = project_path / "images" / "bdv-n5" / "synth_data.n5"
    assert (organell_path / "setup0" / "timepoint0" / "s0").exists()
    assert (organell_path / "setup0" / "timepoint0" / "s3").exists()


def test_valid_project_init(cebra_project_path):
    """Checks standard construction of the Project object"""

    # With a pathlib.Path object
    project = Project(project_path=cebra_project_path)
    assert project.metadata

    # With a string path
    project = Project(project_path=str(cebra_project_path))
    assert project.metadata


def test_invalid_project_init(tmp_path):
    """Checks invalid construction of the Project class"""

    with pytest.raises(FileNotFoundError):
        project = Project(project_path=tmp_path)


# def test_project_clipping(cebra_project_path):
#     """Checks that the clipping values are correctly validated"""

#     # A correct clipping
#     clip = ((0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
#     project = Project(project_path=cebra_project_path, clipping=clip)
#     assert project.clipping == clip

#     # Incorrect clippings throw
#     with pytest.raises(ValueError):
#         project = Project(project_path=cebra_project_path, clipping=(0.2, 0.2, 0.2))
#     with pytest.raises(ValueError):
#         project = Project(
#             project_path=cebra_project_path, clipping=((0.2, 0.2), (0.8, 0.8))
#         )


def test_add_source(cebra_project):
    """Check adding source to a given project"""
    print(cebra_project.available_sources())
    source_dict = {"synth_data": "mito", "synth_data": "unknown"}
    for source, oid in source_dict.items():
        if oid == "unknown":
            with pytest.raises(ValueError):
                cebra_project.add_source(source=source, organelle=oid)
            continue

        cebra_project.add_source(source=source, organelle=oid)

        # Check that we actually added organelles
        assert cebra_project.organelles(ids=f"{oid}_*", return_ids=True)

    # Wrong organelle/source identifier should lead to error
    with pytest.raises(ValueError):
        cebra_project.add_source("wrong_source", "mito")

    with pytest.raises(ValueError):
        cebra_project.add_source("correct_source_todo", "wrong_organelle")


def test_compression_level(cebra_project):
    """Check the reading/writing of the compression level"""

    p = cebra_project

    # Default compression level should be 0
    assert p.compression_level == 0

    # test impossible compression without added source

    p.compression_level = 42
    with pytest.raises(ValueError):
        p.add_source(source="synth_data", organelle="mito")

    # add source
    p.compression_level = 2
    p.add_source(source="synth_data", organelle="mito")

    # compression level shoulnd't be changed after adding source
    assert p.compression_level == 2

    # Change it to 1
    p.compression_level = 1
    assert p.compression_level == 1

    # change it to 42
    with pytest.raises(ValueError):
        p.compression_level = 42


def test_project_organelles(cebra_project_with_sources):
    """Check that the filtering of organelles works correctly"""

    p = cebra_project_with_sources

    for o in p.organelles("m*"):
        assert isinstance(o, Organelle)
        assert o.id.startswith("m")

    assert p.organelles("m*", return_ids=True)


def test_project_sources_data(cebra_project_with_sources):
    p = cebra_project_with_sources

    assert p._sources["synth_data"].data
    assert p._sources["synth_data"].data_resolution
    assert p._sources["synth_data"].resolution
