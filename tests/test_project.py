from organelle_morphology.project import Project
from organelle_morphology.organelle import Organelle
from organelle_morphology.source import DataSource
from .synthetic_data_generator import generate_synthetic_dataset

import pytest
import pathlib
import tempfile
import numpy as np
import trimesh
import trimesh.visual


def test_synthetic_data_generation(
    synthetic_data, cebra_project_path, cebra_project_original_meshes
):
    """Check the synthetic data generation"""

    generate_synthetic_dataset()

    # test data from custom dir
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    project_path_custom, original_meshes_custom = generate_synthetic_dataset(
        working_dir=tmp_dir, n_objects=30, object_size=20, object_distance=100, seed=42
    )
    # test default data
    project_path, original_meshes = synthetic_data

    assert cebra_project_path
    assert cebra_project_original_meshes == original_meshes

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


def test_valid_project_init(cebra_project_path):
    """Checks standard construction of the Project object"""

    # With a pathlib.Path object
    project = Project(project_path=cebra_project_path)
    assert project.path == cebra_project_path

    # With a string path
    project = Project(project_path=str(cebra_project_path))
    assert project.path == cebra_project_path


def test_project_clipping(cebra_project_path):
    """Checks that the clipping values are correctly validated"""

    # A correct clipping
    clip = ((0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
    project = Project(project_path=cebra_project_path, clipping=clip)
    assert np.all(project.clipping == clip)

    # Incorrect clippings throw
    with pytest.raises(ValueError):
        Project(project_path=cebra_project_path, clipping=(0.2, 0.2, 0.2))

    with pytest.raises(ValueError):
        Project(project_path=cebra_project_path, clipping=((0.2, 0.2), (0.8, 0.8)))

    with pytest.raises(ValueError):
        Project(
            project_path=cebra_project_path,
            clipping=((0.2, 0.6, 0.2), (0.8, 0.5, 0.8)),
        )
    with pytest.raises(ValueError):
        Project(
            project_path=cebra_project_path,
            clipping=((-0.2, 0.2, 0.2), (0.8, 0.5, 0.8)),
        )
    with pytest.raises(ValueError):
        Project(
            project_path=cebra_project_path,
            clipping=((0.2, 0.2, 0.2), (0.8, 1.5, 0.8)),
        )
    # add a source and check that the clipping is correctly propagated
    project.add_source(xml_path="synth_data", organelle="mito")
    assert list(project.sources.values())[0].data.shape == (141, 144, 198)


def test_add_source(cebra_project):
    """Check adding source to a given project"""
    source_dict = {"mito": "synth_data", "unknown": "synth_data"}
    for oid, source in source_dict.items():
        if oid == "unknown":
            with pytest.raises(ValueError):
                cebra_project.add_source(xml_path=source, organelle=oid)
            continue
        else:
            cebra_project.add_source(xml_path=source, organelle=oid)
            # Check that we actually added organelles
            assert cebra_project.organelles(ids=f"{oid}_*")


def test_add_source_wrong_source(cebra_project):
    with pytest.raises(FileNotFoundError):
        cebra_project.add_source("wrong_source", "mito")


def test_project_organelles(cebra_project_with_sources):
    """Check that the filtering of organelles works correctly"""

    p = cebra_project_with_sources

    for o, id in zip(p.organelles("m*"), p.organelle_ids("m*")):
        assert isinstance(o, Organelle)
        assert o.id.startswith("m")


def test_project_sources_data(cebra_project_with_sources):
    p = cebra_project_with_sources

    assert isinstance(p.sources["synth_data"], DataSource)
    assert p.sources["synth_data"].org_name == "mito"
    assert p.sources["synth_data"].background_label == 0


def test_geometric_properties(
    cebra_project_with_sources, cebra_project_original_meshes
):
    p = cebra_project_with_sources

    assert len(p.geometric_properties) == len(cebra_project_original_meshes)

    for org_key in p.geometric_properties.index:
        geometric_properties = p.geometric_properties.loc[org_key]
        print(org_key)
        mesh_id = int(org_key.split("_")[-1])

        # skip these (see test_geometric_data)
        if mesh_id in [9, 17]:
            continue

        original_mesh = cebra_project_original_meshes[mesh_id]

        assert np.isclose(
            original_mesh["volume"],
            geometric_properties["voxel_volume"],
            rtol=0.25,
            atol=500,
        )
        assert np.isclose(
            original_mesh["area"],
            geometric_properties["mesh_area"],
            rtol=0.40,
            atol=100,
        )


def test_morphology_map(cebra_project_with_sources):
    p = cebra_project_with_sources

    assert p.morphology_map
    assert len(p.morphology_map["synth_data"]) == 19


def test_distance_matrix(cebra_project_with_sources):
    p = cebra_project_with_sources

    assert p.distance_matrix.shape == (19, 19)


def test_skeletonize(cebra_project_with_sources):
    p = cebra_project_with_sources

    filter_id = p.organelles("m*")[0].id

    p.skeletonize_vertex_clusters(filter_id)
    assert p.organelles(filter_id)[0].skeleton.method == "vertex_clusters"

    p.skeletonize_wavefront(skip_existing=True)

    assert p.organelles(filter_id)[0].skeleton.method == "vertex_clusters"

    p.skeletonize_wavefront(skip_existing=False)
    assert p.organelles(filter_id)[0].skeleton.method == "wavefront"

    assert p.skeleton_info.shape == (19, 9)


def test_show_mesh_scene(cebra_project_with_sources):
    p = cebra_project_with_sources
    meshes = p._meshes

    scene = trimesh.Scene()
    for source_key in meshes.keys():
        for org_key in meshes[source_key].keys():
            mesh = meshes[source_key][org_key]
            mesh.visual.face_colors = trimesh.visual.random_color()
            scene.add_geometry(mesh)
    # scene.show()  # don't run this on ci


def test_mcs(cebra_project_with_sources):
    p = cebra_project_with_sources

    ids = "*"
    p.search_mcs(
        "far_contacts",
        min_distance=0.1,
        max_distance=0.5,
    )

    p.search_mcs(
        "close_contacts",
        min_distance=0.0,
        max_distance=0.10,
    )

    mcs_properties = p.get_mcs_properties(ids=ids)

    with pytest.raises(KeyError):
        mcs_properties.loc["far_contacts"]

    assert len(mcs_properties) == 2

    with pytest.raises(ValueError):
        p.search_mcs(
            "far_contacts",
            min_distance=0.1,
            max_distance=10,
        )
        p.search_mcs(
            "far_contacts",
            min_distance=0.1,
            max_distance=10,
            override_mcs_label=True,
        )

    p.search_mcs(
        "very_far_contacts",
        min_distance=0.1,
        max_distance=10,
    )

    mcs_properties = p.get_mcs_properties(ids=ids)
    assert len(mcs_properties) == 13

    print(mcs_properties)

    mcs_overview = p.get_mcs_overview()
    assert mcs_overview.shape == (10, 2)

    mcs_overview = p.get_mcs_overview(mcs_filter="very_far_contacts")
    assert mcs_overview.shape == (10, 1)


def test_show(cebra_project_with_sources):
    p = cebra_project_with_sources

    p.show()

    p.show(ids="*1")

    p.show(show_morphology=True)

    # no skeletons present
    p.show(show_skeleton=True)

    p.skeletonize_wavefront(skip_existing=True)
    p.show(show_skeleton=True)
