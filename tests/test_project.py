from trimesh import Trimesh
from trimesh.path import Path3D
from organelle_morphology.project import Project
from organelle_morphology.statistics import Record
from .synthetic_data_generator import generate_synthetic_dataset

import pytest
import pathlib
import numpy as np


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
    assert project.path.samefile(cebra_project_path)

    # With a string path
    project = Project(project_path=str(cebra_project_path), client=client)
    assert project.path.samefile(cebra_project_path)


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
            assert project.get_organelles(ids=f"{oid}_*")

            with pytest.raises(ValueError, match="Source already loaded!"):
                project.add_source(xml_path=source, organelle=oid)

    assert "synth_data" in project.sources
    s = project.sources["synth_data"]
    assert s.project == project


def test_add_source_wrong_source(synthetic_data, client):
    project = Project(synthetic_data[0], client=client)
    with pytest.raises(FileNotFoundError):
        project.add_source("wrong_source", "mito")


def test_skeletonize_wavefront(mocker, project_with_sources):
    mock_gen_skel = mocker.patch.object(
        project_with_sources.sources["synth_data"], "generate_skeletons"
    )
    mock_gen_skel.return_value = [None]

    project_with_sources.skeletonize_wavefront()
    mock_gen_skel.assert_called_once()


def test_skeletonize_vertex_clusters(mocker, project_with_sources):
    mock_gen_skel = mocker.patch.object(
        project_with_sources.sources["synth_data"], "generate_skeletons"
    )
    mock_gen_skel.return_value = [None]

    project_with_sources.skeletonize_vertex_clusters()
    mock_gen_skel.assert_called_once()


def test_show_plain(project_with_sources, mocker):
    p: Project = project_with_sources

    mock_util_show = mocker.patch("organelle_morphology.project.show")

    p.show()
    mock_util_show.assert_called_once()
    to_show = mock_util_show.call_args[0][0]
    assert len(to_show) == 2
    assert isinstance(to_show[0], Trimesh)
    assert isinstance(to_show[1], Path3D)
    assert 15 < len(np.unique(to_show[0].visual.vertex_colors, axis=0)) < 30


def test_show_skeleton(project_with_sources, mocker):
    p: Project = project_with_sources

    mock_util_show = mocker.patch("organelle_morphology.project.show")

    p.show(skeleton=True)
    mock_util_show.assert_called_once()
    to_show = mock_util_show.call_args[0][0]
    assert len(to_show) == 2
    assert isinstance(to_show[0], Trimesh)
    assert isinstance(to_show[1], Path3D)
    assert np.all(to_show[0].visual.vertex_colors[:, -1] == 100)


def test_show_curvature(project_with_sources, mocker):
    p: Project = project_with_sources
    s = project_with_sources.sources["synth_data"]

    mock_util_show = mocker.patch("organelle_morphology.project.show")
    mock_curvature = mocker.patch.object(s, "get_meshes_curvature_colored")
    mock_curvature.return_value = s.meshes

    p.show(curvature=True)
    mock_curvature.assert_called_once()
    mock_util_show.assert_called_once()
    to_show = mock_util_show.call_args[0][0]
    assert len(to_show) == 2
    assert isinstance(to_show[0], Trimesh)
    assert isinstance(to_show[1], Path3D)


def test_show_highlight(project_with_sources, mocker):
    p: Project = project_with_sources

    mock_util_show = mocker.patch("organelle_morphology.project.show")
    p.show(ids_highlight="Nothing")

    to_show = mock_util_show.call_args[0][0]
    colors = np.unique(to_show[0].visual.vertex_colors, axis=0)
    assert len(colors) == 1

    p.show(ids_highlight="mito*")
    to_show = mock_util_show.call_args[0][0]
    colors2 = np.unique(to_show[0].visual.vertex_colors, axis=0)
    assert len(colors2) == 1
    assert np.any(colors[0] != colors2[0])


def test_show_domain_box_off(project_with_sources, mocker):
    p: Project = project_with_sources

    mock_util_show = mocker.patch("organelle_morphology.project.show")
    p.show(domain_box=False)

    to_show = mock_util_show.call_args[0][0]
    assert len(to_show) == 1
    assert isinstance(to_show[0], Trimesh)


def test_show_box(project_with_sources, mocker):
    p: Project = project_with_sources

    mock_util_show = mocker.patch("organelle_morphology.project.show")
    p.show(box=((0, 0, 0), (0.5, 0.5, 0.5)))

    to_show = mock_util_show.call_args[0][0]
    assert len(to_show) == 3
    assert isinstance(to_show[0], Trimesh)
    assert isinstance(to_show[1], Path3D)
    assert isinstance(to_show[2], Path3D)


def test_show_clipping_box(project_with_sources, mocker):
    p: Project = project_with_sources

    mock_util_show = mocker.patch("organelle_morphology.project.show")
    p.clipping = ((0, 0, 0), (0.5, 0.5, 0.5))
    p.show()

    to_show = mock_util_show.call_args[0][0]
    assert len(to_show) == 3
    assert isinstance(to_show[0], Trimesh)
    assert isinstance(to_show[1], Path3D)
    assert isinstance(to_show[2], Path3D)


def test_recreate_client(project):
    project.client = None
    project.recreate_client()
    assert project.client


def test_project_str(project):
    assert str(project)


def test_set_loglevel(project):
    with pytest.raises(AttributeError):
        project.set_loglevel("a")
    project.set_loglevel("CRITICAL")


def test_path(project):
    p = project.path
    assert isinstance(p, pathlib.Path)
    assert p.is_absolute()


def test_cache_settings(project_with_sources):
    cs = project_with_sources.cache_settings
    assert isinstance(cs, dict)


def test_curvature_map(project_with_sources, mocker):
    p: Project = project_with_sources
    s = project_with_sources.sources["synth_data"]
    mock_curv = mocker.patch.object(s, "calc_curvature")

    assert p.curvature_map == {"synth_data": {}}
    mock_curv.assert_called_once()


def test_clipping(project):
    assert project.clipping is None
    project._clipping = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    assert project.clipping == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))


def test_set_clippign_wrong(project):
    with pytest.raises(ValueError):
        project.clipping = ((0.6, 0.0, 0.0), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        project.clipping = ((0.0, 0.6, 0.0), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        project.clipping = ((0.0, 0.0, 0.6), (0.5, 0.5, 0.0))

    with pytest.raises(ValueError):
        project.clipping = ((-0.1, 0.0, 0.6), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        project.clipping = ((0.1, 0.0, 0.6), (0.5, 0.5, 1.5))

    with pytest.raises(ValueError):
        project.clipping = ((0.1, 0.0, 0.6), (0.5, 0.5, 0.5, 0.4))
    with pytest.raises(ValueError):
        project.clipping = ((0.1, 0.0), (0.5, 0.5, 0.4))


def test_set_clipping(project_with_sources, mocker):
    p = project_with_sources
    s = project_with_sources.sources["synth_data"]
    mock_clear_cache = mocker.patch.object(s, "clear_memory_cache")
    p.clipping = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))

    assert np.all(p.clipping == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
    assert isinstance(p.clipping, np.ndarray)
    mock_clear_cache.assert_called_once()


def test_organelles(project_with_sources):
    orgs = project_with_sources.organelles
    assert len(orgs) == 19


def test_organelle_ids(project_with_sources):
    org_ids = project_with_sources.organelle_ids
    assert len(org_ids) == 19


def test_get_organelles(project_with_sources, mocker):
    p = project_with_sources
    s = project_with_sources.sources["synth_data"]
    mock_getter = mocker.patch.object(s, "get_organelles")

    p.get_organelles()
    mock_getter.assert_called_once()

    mock_getter.reset_mock()
    p.get_organelles("")
    mock_getter.assert_called_once()

    mock_getter.reset_mock()
    p.get_organelles([""])
    mock_getter.assert_called_once()

    mock_getter.reset_mock()
    p.get_organelles(["mito*", "er*"])
    assert mock_getter.call_count == 2

    mock_getter.reset_mock()
    p.permanent_blacklist = ["mito*"]
    p.get_organelles("mito*")
    mock_getter.assert_not_called()


def test_get_organelle_ids(project_with_sources, mocker):
    p = project_with_sources
    s = project_with_sources.sources["synth_data"]
    mock_getter = mocker.patch.object(s, "get_organelle_ids")

    p.get_organelle_ids()
    mock_getter.assert_called_once()

    mock_getter.reset_mock()
    p.get_organelle_ids("")
    mock_getter.assert_called_once()

    mock_getter.reset_mock()
    p.get_organelle_ids([""])
    mock_getter.assert_called_once()

    mock_getter.reset_mock()
    p.get_organelle_ids(["mito*", "er*"])
    assert mock_getter.call_count == 2

    mock_getter.reset_mock()
    p.permanent_blacklist = ["mito*"]
    p.get_organelle_ids("mito*")
    mock_getter.assert_not_called()


def test_clear_caches(project_with_sources, mocker):
    p = project_with_sources
    s = project_with_sources.sources["synth_data"]
    s_cache = s.cache
    s_cache["test"] = "test_content"
    s_cache.clear_memory_cache()
    assert s._cache is not None
    mock_source_reset = mocker.spy(s, "clear_memory_cache")

    p.clear_caches()
    mock_source_reset.assert_called_once()
    assert s._cache is None
    # disk cache still there, populates mem cache
    assert s_cache["test"] == "test_content"

    p.clear_caches(clear_disk=True)
    assert "test" in s_cache
    assert "test" not in s.cache


def test_stat_stats(project, mocker):
    assert project.get_stat_stats() == {}

    stat = Record(mocker.sentinel, mocker.sentinel)
    project.add_stat(stat)

    assert project.get_stat_stats()["_Sentinel"] == 1
