from organelle_morphology.distance_calculations import (
    MembraneContactSiteCalculator,
    generate_distance_matrix,
    generate_mcs,
    get_min_dist,
)
from pandas import DataFrame
import pytest
import numpy as np
import trimesh


def test_distance_matrix(project_with_sources):
    project_with_sources.max_distance = 100
    df = generate_distance_matrix(project_with_sources)
    assert isinstance(df, DataFrame)
    assert df.shape == (19, 19)
    for ind, row in df.iterrows():
        assert row.loc[df.index[3]] == df.loc[df.index[3], ind]
        assert row.pop(ind) == -1.0
        assert np.all(row >= 0.0)


def test_distance_matrix_incomplete(project_with_sources):
    project_with_sources.max_distance = 1
    df = generate_distance_matrix(project_with_sources)
    assert isinstance(df, DataFrame)
    assert df.shape == (19, 19)
    assert (df < 0).sum().sum() == 341
    for ind, row in df.iterrows():
        assert row.loc[df.index[3]] == df.loc[df.index[3], ind]
        assert row.pop(ind) == -1.0


def test_distance_matrix_no_source(project):
    with pytest.raises(ValueError):
        generate_distance_matrix(project)


def test_distance_matrix_cached(project_with_sources):
    dummy = "DUMMY"
    project_with_sources.cache["distance_matrix"] = dummy
    project_with_sources.cache["max_distance_computed"] = 100

    assert generate_distance_matrix(project_with_sources) is dummy


def test_get_min_dist():
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    points_touching = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh1 = trimesh.Trimesh(vertices=points, faces=faces)
    mesh2 = trimesh.Trimesh(vertices=points + [0, 0, 2], faces=faces)
    mesh3 = trimesh.Trimesh(vertices=points_touching, faces=faces)

    d1 = get_min_dist(("a", "b", mesh1, mesh2))
    d2 = get_min_dist(("a", "b", mesh1, mesh1))
    d3 = get_min_dist(("a", "b", mesh1, mesh3))

    assert all([r[0] == ("a", "b") for r in [d1, d2, d3]])
    assert d1[1] == 2.0
    assert d2[1] == 0.0
    assert d3[1] == 0.0


def test_search_mcs(project_with_sources):
    p = project_with_sources
    s = p.sources["synth_data"]
    meshes = [s.meshes[i].compute() for i in (7, 3)]

    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", meshes[0], meshes[1])

    assert calculator.id_source == "a"
    assert calculator.id_target == "b"
    assert calculator.distances.shape == (226,)
    assert all(calculator.dot_products < 0)
    np.testing.assert_almost_equal(calculator.min_distance, 88.97752525)

    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", meshes[1], meshes[0])

    assert calculator.id_source == "b"
    assert calculator.id_target == "a"
    np.testing.assert_almost_equal(calculator.min_distance, 88.97752525)


def test_search_mcs_inverted_normals(project_with_sources, mocker):
    p = project_with_sources
    s = p.sources["synth_data"]
    meshes = [s.meshes[i].compute() for i in (7, 3)]

    # test inverted normals
    calculator = MembraneContactSiteCalculator()
    meshes[1].faces = meshes[1].faces[:, [2, 1, 0]]

    mocker.patch.object(calculator, "_repair_meshes")
    calculator.search_mcs("a", "b", meshes[0], meshes[1])

    assert calculator.id_source == "a"
    assert calculator.id_target == "b"
    assert calculator.distances.shape == (226,)
    assert all(calculator.dot_products > 0)
    np.testing.assert_almost_equal(calculator.min_distance, 88.97752525216691)


def test_analyze_mcs(project_with_sources):
    p = project_with_sources
    s = p.sources["synth_data"]
    meshes = [s.meshes[i].compute() for i in (7, 3)]

    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", meshes[0], meshes[1])

    calculator.analyze_mcs("test", 90, 0)
    assert np.all(calculator._distances < 90)
    assert np.all(calculator._distances > 88)
    assert calculator._distances.shape == (20,)
    assert calculator.distances.shape == (226,)

    assert isinstance(calculator.mcs_source, dict)
    assert isinstance(calculator.mcs_target, dict)

    source_v = calculator.mesh_source.vertices[calculator.mcs_source["vertices_index"]]
    target_v = calculator.mesh_target.vertices[calculator.mcs_target["vertices_index"]]
    np.testing.assert_array_equal(
        calculator._distances, np.linalg.norm(source_v - target_v, axis=1)
    )


def test_search_mcs_watertight_meshes():
    """Test the search_mcs function with different combinations of watertight meshes."""

    # Create simple watertight mesh (sphere)
    mesh_watertight = trimesh.primitives.Sphere(radius=1.0)

    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],  # Triangle 1
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],  # Triangle 2
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],  # First triangle
            [3, 4, 5],  # Second triangle (disconnected)
        ]
    )
    mesh_non_watertight = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Test case 1: Both meshes watertight
    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", mesh_watertight, mesh_watertight)

    assert calculator.id_source == "a"
    assert calculator.id_target == "b"
    assert calculator.distances is not None
    assert calculator.min_distance >= 0 or np.isnan(calculator.min_distance)

    # Test case 2: First mesh watertight, second not
    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", mesh_watertight, mesh_non_watertight)

    assert calculator.id_source == "a"
    assert calculator.id_target == "b"
    assert calculator.distances is not None

    # Test case 3: First mesh not watertight, second watertight
    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", mesh_non_watertight, mesh_watertight)

    assert calculator.id_source == "b"
    assert calculator.id_target == "a"
    assert calculator.distances is not None

    # Test case 4: Neither mesh watertight
    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", mesh_non_watertight, mesh_non_watertight)

    assert calculator.id_source == "a"  # Should pick smaller mesh as source
    assert calculator.id_target == "b"
    assert calculator.distances is not None

    # Test with different vertex counts to verify ordering logic
    small_mesh = trimesh.primitives.Sphere(radius=1.0, subdivisions=3)
    large_mesh = trimesh.primitives.Sphere(radius=1.0, subdivisions=5)

    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("small", "large", small_mesh, large_mesh)
    assert calculator.id_source == "large"
    assert calculator.id_target == "small"


def test_generate_mcs(project_with_sources):
    p = project_with_sources

    generate_mcs(p, "*", "*", max_distance=90, min_distance=0)
    assert list(p.mcs_labels)[0] == "0-90,-"
    org = p.get_organelles("mito_0007")[0]
    assert "0-90,-" in org.mcs.keys()
    # deviation due to mesh normals and subsequent filtering
    # -> different area on windows

    mcs_stats = p.registry.get_by_type("McsData")
    assert len(mcs_stats) == 19
    stat = [s for s in mcs_stats if s.meta.organelle_id == org.id][0]

    np.testing.assert_almost_equal(stat.data.mean_area, 240.6018779, decimal=1)
    assert stat.data.n_contacts == 7
