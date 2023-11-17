from organelle_morphology.project import *
from organelle_morphology.organelle import Organelle

import pytest


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


def test_project_clipping(cebra_project_path):
    """Checks that the clipping values are correctly validated"""

    # A correct clipping
    clip = ((0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
    project = Project(project_path=cebra_project_path, clipping=clip)
    assert project.clipping == clip

    # Incorrect clippings throw
    with pytest.raises(ValueError):
        project = Project(project_path=cebra_project_path, clipping=(0.2, 0.2, 0.2))
    with pytest.raises(ValueError):
        project = Project(
            project_path=cebra_project_path, clipping=((0.2, 0.2), (0.8, 0.8))
        )


def test_add_source(cebra_project):
    """Check adding source to a given project"""

    # TODO: Get the pairs for the test data here
    for oid, source in dict().items():
        cebra_project.add_source(source=source, organelle=oid)

        # Check that we actually added organelles
        assert cebra_project.organelles(ids=f"{oid}_*", return_ids=True)

    # Wrong organelle/source identifier should lead to error
    with pytest.raises(ValueError):
        cebra_project.add_source("wrong_source", "mito")

    with pytest.raises(ValueError):
        cebra_project.add_source("correct_source_todo", "wrong_organelle")


def test_compression_level(cebra_project_with_sources):
    """Check the reading/writing of the compression level"""

    p = cebra_project_with_sources

    # Default compression level should be 0
    assert p.compression_level == 0

    # Change it to 1
    p.compression_level = 1
    assert p.compression_level == 1


def test_project_organelles(cebra_project_with_sources):
    """Check that the filtering of organelles works correctly"""

    p = cebra_project_with_sources

    for o in p.organelles("m*"):
        assert isinstance(o, Organelle)
        assert o.id.startswith("m")

    assert p.organelles("m*", return_ids=True)
