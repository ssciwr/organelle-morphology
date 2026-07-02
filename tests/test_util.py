import trimesh
from organelle_morphology.util import (
    block_to_coords,
    get_neighboring_chunks,
    measure_gaussian_curvature_delayed,
    simplify_mesh,
)
import numpy as np
import dask.array as da


def test_measure_curvature_parametrized(project_with_sources):
    project_with_sources.simplify = 0.0
    s = project_with_sources.sources["synth_data"]
    mesh = s.meshes[1]
    delayed = measure_gaussian_curvature_delayed(tmesh=mesh, radius=1)
    curv = delayed.compute()
    assert curv.shape == (866,)


def test_central_index_returns_26_neighbors():
    shape = (5, 5, 5)
    index = (2, 2, 2)  # Center
    neighbors = get_neighboring_chunks(index, shape)

    assert len(neighbors) == 26, f"Expected 26 neighbors, got {len(neighbors)}"

    # Verify all neighbors are unique
    assert len(set(neighbors)) == 26, "Neighbors should all be unique"

    # Verify none of the neighbors are the center itself
    assert index not in neighbors, "Neighbors should not include the center index"

    # Verify all indices are within bounds
    for nx, ny, nz in neighbors:
        assert 0 <= nx < shape[0], f"nx={nx} out of bounds"
        assert 0 <= ny < shape[1], f"ny={ny} out of bounds"
        assert 0 <= nz < shape[2], f"nz={nz} out of bounds"


def test_corner_index_returns_7_neighbors():
    shape = (5, 5, 5)
    corners = [
        (0, 0, 0),
        (0, 0, 4),
        (0, 4, 0),
        (0, 4, 4),
        (4, 0, 0),
        (4, 0, 4),
        (4, 4, 0),
        (4, 4, 4),
    ]

    for corner in corners:
        neighbors = get_neighboring_chunks(corner, shape)
        assert len(neighbors) == 7, (
            f"Corner {corner} should have 7 neighbors, got {len(neighbors)}"
        )

        # Verify all neighbors are within bounds
        for nx, ny, nz in neighbors:
            assert 0 <= nx < shape[0]
            assert 0 <= ny < shape[1]
            assert 0 <= nz < shape[2]

        # Verify center is not included
        assert corner not in neighbors


def test_edge_index_returns_11_neighbors():
    shape = (5, 5, 5)
    edge_index = (0, 0, 2)  # On one face, not a corner
    neighbors = get_neighboring_chunks(edge_index, shape)

    assert len(neighbors) == 11, (
        f"Edge index should have 11 neighbors, got {len(neighbors)}"
    )


def test_face_index_returns_17_neighbors():
    shape = (5, 5, 5)
    face_index = (0, 2, 2)  # On edge between two faces
    neighbors = get_neighboring_chunks(face_index, shape)

    assert len(neighbors) == 17, (
        f"Face index should have 17 neighbors, got {len(neighbors)}"
    )


def test_small_matrix_1x1x1():
    shape = (1, 1, 1)
    index = (0, 0, 0)
    neighbors = get_neighboring_chunks(index, shape)

    assert len(neighbors) == 0, (
        f"1x1x1 matrix should have 0 neighbors, got {len(neighbors)}"
    )


def test_block_to_coords():
    data = da.from_array(np.empty((100, 60, 20)), chunks=18)
    last_corner = [-1, -1, -1]
    for block in np.ndindex(data.blocks.shape):
        lc, uc = block_to_coords(block, (0.03, 0.02, 0.1), data, [10, 20, 30])
        assert np.any([c > last_c for c, last_c in zip(lc, last_corner)])
        assert len(lc) == 3
        assert len(uc) == 3
        assert np.all(lc < uc)
        assert lc[0] >= 10
        assert lc[1] >= 20
        assert lc[2] >= 30
        last_corner = lc


def test_simplify_mesh_cylider():
    mesh = trimesh.creation.cylinder(radius=2, height=5)
    assert mesh.vertices.shape == (66, 3)

    simplified = simplify_mesh(mesh, factor=0.1)
    assert simplified.vertices.shape == (64, 3)

    simplified = simplify_mesh(mesh, factor=0.5)
    assert simplified.vertices.shape == (34, 3)

    simplified = simplify_mesh(mesh, factor=0.8)
    assert simplified.vertices.shape == (14, 3)


def test_simplify_mesh_small_cylider():
    # small mesh does not get simplified
    mesh = trimesh.creation.cylinder(radius=2, height=5, sections=3)
    assert mesh.vertices.shape == (8, 3)
    assert simplify_mesh(mesh, factor=0.5) == mesh


def test_simplify_mesh_icosphere_aggression(mocker):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2)
    spy = mocker.spy(mesh, "simplify_quadric_decimation")
    assert mesh.vertices.shape == (42, 3)

    simplified = simplify_mesh(mesh, factor=0.5, aggression=1)
    assert simplified.vertices.shape == (42, 3)
    assert spy.call_count == 10
    simplified = simplify_mesh(mesh, factor=0.5, aggression=7)
    assert simplified.vertices.shape == (22, 3)
    assert spy.call_count == 11
    simplified = simplify_mesh(mesh, factor=0.5, aggression=None)
    assert simplified.vertices.shape == (22, 3)
    assert spy.call_count == 16
