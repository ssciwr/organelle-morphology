from organelle_morphology.project import *
from organelle_morphology.organelle import Organelle
from .synthetic_data_generator import generate_synthetic_dataset

import pytest
import pathlib
import tempfile
import numpy as np


def test_synthetic_data_generation(
    synthetic_data, cebra_project_path, cebra_project_original_meshes
):
    """Check that the synthetic data is correctly generated"""

    generate_synthetic_dataset()

    # test data from custom dir
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    project_path_custom, original_meshes_custom = generate_synthetic_dataset(
        working_dir=tmp_dir, seed=42, n_objects=10, object_size=10, object_distance=20
    )

    # test default data
    project_path, original_meshes = synthetic_data

    assert cebra_project_path
    assert cebra_project_original_meshes == original_meshes

    # assert custom meshes and original meshes are the same.
    # for this we will only compare the area of the meshes
    for mesh_custom, mesh_orig in zip(original_meshes_custom, original_meshes):
        assert np.isclose(mesh_custom["area"], mesh_orig["area"])

    project_path = pathlib.Path(project_path)
    for p_path in [project_path, project_path_custom]:
        assert (p_path / "project.json").exists()
        assert (p_path / "CebraEM" / "dataset.json").exists()
        assert (p_path / "CebraEM" / "images" / "bdv-n5" / "synth_data.n5").exists()
        assert (p_path / "CebraEM" / "images" / "bdv-n5" / "synth_data.xml").exists()

        organell_path = p_path / "CebraEM" / "images" / "bdv-n5" / "synth_data.n5"
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
    source_dict = {"mito": "synth_data", "unknown": "synth_data"}
    for oid, source in source_dict.items():
        if oid == "unknown":
            with pytest.raises(ValueError):
                cebra_project.add_source(source=source, organelle=oid)
            continue
        else:
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

    # change to -1
    with pytest.raises(ValueError):
        p.compression_level = -1


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
