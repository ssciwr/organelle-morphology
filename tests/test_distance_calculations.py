from organelle_morphology.distance_calculations import (
    MembraneContactSiteCalculator,
    generate_distance_matrix,
    generate_mcs,
    get_min_dist,
)
from pandas import DataFrame
import numpy as np
import trimesh


def test_distance_matrix(project_with_sources):
    df = generate_distance_matrix(project_with_sources)
    assert isinstance(df, DataFrame)
    assert df.shape == (19, 19)
    for ind, row in df.iterrows():
        assert row[ind] == 0.0
        assert row.loc[df.index[3]] == df.loc[df.index[3], ind]
    assert all(df >= 0.0)


def test_distance_matrix_cached(project):
    dummy = "DUMMY"
    project.cache["distance_matrix"] = dummy

    assert generate_distance_matrix(project) is dummy


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
    meshes = [m.compute() for m in s.meshes.values()]

    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", meshes[0], meshes[1])

    assert calculator.id_source == "a"
    assert calculator.id_target == "b"
    assert calculator.distances.shape == (226,)
    assert all(calculator.dot_products < 0)
    np.testing.assert_almost_equal(calculator.min_distance, 88.97752525216691)

    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", meshes[1], meshes[0])

    assert calculator.id_source == "b"
    assert calculator.id_target == "a"
    np.testing.assert_almost_equal(calculator.min_distance, 88.97752525216691)


def test_search_mcs_inverted_normals(project_with_sources, mocker):
    p = project_with_sources
    s = p.sources["synth_data"]
    meshes = [m.compute() for m in s.meshes.values()]

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
    meshes = [m.compute() for m in s.meshes.values()]

    calculator = MembraneContactSiteCalculator()
    calculator.search_mcs("a", "b", meshes[0], meshes[1])

    calculator.analyze_mcs(90, 0)
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


def test_generate_mcs(project_with_sources):
    p = project_with_sources

    generate_mcs(p, 90)
    # FIXME: implement test
