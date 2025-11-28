from organelle_morphology.project import Project
from .synthetic_data_generator import generate_synthetic_dataset

from dask.distributed import LocalCluster

import pytest
import pathlib
import numpy as np


@pytest.fixture(scope="session")
def client():
    cluster = LocalCluster()
    yield cluster.get_client()
    cluster.close()


def test_synthetic_data_generation(synthetic_data, tmp_path):
    """Check the synthetic data generation"""

    generate_synthetic_dataset()

    # test data from custom dir
    project_path_custom, original_meshes_custom = generate_synthetic_dataset(
        working_dir=tmp_path, n_objects=30, object_size=20, object_distance=100, seed=42
    )
    # test default data
    project_path, original_meshes = synthetic_data

    # assert custom meshes and original meshes are the same.
    # for this we will only compare the area of the meshes
    for mesh_custom, mesh_orig in zip(
        original_meshes_custom.values(), original_meshes.values()
    ):
        assert np.isclose(mesh_custom["area"], mesh_orig["area"])

    project_path = pathlib.Path(project_path)
    for p_path in [project_path, project_path_custom]:
        assert (p_path / "synth_data.n5").exists()
        assert (p_path / "synth_data.xml").exists()

        organell_path = p_path / "synth_data.n5"
        assert (organell_path / "setup0" / "timepoint0" / "s0").exists()
        assert (organell_path / "setup0" / "timepoint0" / "s3").exists()


def test_valid_project_init(synthetic_data, client):
    """Checks standard construction of the Project object"""
    cebra_project_path = synthetic_data[0]
    # With a pathlib.Path object
    project = Project(project_path=cebra_project_path, client=client)
    assert project.path == cebra_project_path

    # With a string path
    project = Project(project_path=str(cebra_project_path), client=client)
    assert project.path == cebra_project_path


def test_project_clipping(synthetic_data, client):
    """Checks that the clipping values are correctly validated"""
    cebra_project_path = synthetic_data[0]

    # A correct clipping
    clip = ((0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
    project = Project(project_path=cebra_project_path, clipping=clip, client=client)
    assert np.all(project.clipping == clip)

    # Incorrect clippings throw

    with pytest.raises(ValueError):
        Project(
            project_path=cebra_project_path,
            clipping=((0.2, 0.2), (0.8, 0.8)),
            client=client,
        )

    with pytest.raises(ValueError):
        Project(
            project_path=cebra_project_path,
            clipping=((0.2, 0.6, 0.2), (0.8, 0.5, 0.8)),
            client=client,
        )
    with pytest.raises(ValueError):
        Project(
            project_path=cebra_project_path,
            clipping=((-0.2, 0.2, 0.2), (0.8, 0.5, 0.8)),
            client=client,
        )
    with pytest.raises(ValueError):
        Project(
            project_path=cebra_project_path,
            clipping=((0.2, 0.2, 0.2), (0.8, 1.5, 0.8)),
            client=client,
        )
    # add a source and check that the clipping is correctly propagated
    project.add_source(xml_path="synth_data", organelle="mito")
    assert list(project.sources.values())[0].data.shape == (141, 144, 198)


def test_add_source(synthetic_data, client):
    """Check adding source to a given project"""
    project = Project(synthetic_data[0], client=client)
    source_dict = {"mito": "synth_data", "unknown": "synth_data"}
    for oid, source in source_dict.items():
        if oid == "unknown":
            with pytest.raises(ValueError):
                project.add_source(xml_path=source, organelle=oid)
            continue
        else:
            project.add_source(xml_path=source, organelle=oid)
            # Check that we actually added organelles
            assert project.organelles(ids=f"{oid}_*")

            with pytest.raises(ValueError, match="Source already loaded!"):
                project.add_source(xml_path=source, organelle=oid)

    assert "synth_data" in project.sources
    s = project.sources["synth_data"]
    assert s.project == project


def test_add_source_wrong_source(synthetic_data, client):
    project = Project(synthetic_data[0], client=client)
    with pytest.raises(FileNotFoundError):
        project.add_source("wrong_source", "mito")
