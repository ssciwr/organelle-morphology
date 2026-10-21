import trimesh
from organelle_morphology.util import (
    block_to_coords,
    get_neighboring_chunks,
    measure_gaussian_curvature_delayed,
    merge_mesh_dict_values,
    merge_meshes,
    simplify_mesh,
)
import numpy as np
import dask.array as da
import pytest

from .mesh_asserts import assert_watertight


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


def _voxel_fragment(origin, open_side, resolution):
    """Surface mesh of one voxel cube at the given voxel origin.

    Mimics the per-chunk fragments of `block_mesher`: vertices sit on the
    half-voxel grid (resolution * (k + 0.5)) and the fragment is open on the
    side where the neighboring chunk fragment was cut away. Adjacent
    fragments meet with bitwise-equal seam vertices.
    """
    base = np.asarray(origin, dtype=np.int64)
    corners = np.array(
        [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64
    )
    verts = (base + corners + 0.5) * resolution
    idx = {(i, j, k): n for n, (i, j, k) in enumerate(map(tuple, corners))}
    quads = {
        "-x": ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
        "+x": ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
        "-y": ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
        "+y": ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
        "-z": ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
        "+z": ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
    }
    faces = []
    for side, (a, b, c, d) in quads.items():
        if side != open_side:
            faces.extend(([idx[a], idx[b], idx[c]], [idx[a], idx[c], idx[d]]))
    return trimesh.Trimesh(verts, np.array(faces), process=False)


def _merge_mesh_dict_values(meshes, **kwargs):
    """Take a list like merge_meshes, for parametrization"""
    return merge_mesh_dict_values(dict(enumerate(meshes)), **kwargs)


@pytest.mark.parametrize("merge", [merge_meshes, _merge_mesh_dict_values])
@pytest.mark.parametrize("resolution", [0.01, 0.005])
@pytest.mark.parametrize("weld_units", ["um", "voxel"])
def test_merge_fragments_watertight_at_fine_resolution(merge, resolution, weld_units):
    """Regression test for #20: merged chunk fragments must be watertight.

    At fine physical resolutions (e.g. s0/s1 of the example dataset) mesh
    vertices are spaced by half a voxel, which is finer than a fixed 0.01 um
    weld grid: welding with `digits_vertex=2` on micrometer coordinates
    collapses distinct vertices and creates degenerate faces and
    non-manifold edges. Welding on a 0.01-voxel grid (or exactly) closes
    the seam — whose vertices are bitwise equal — without touching
    anything else.
    """
    fragments = [
        _voxel_fragment((0, 0, 0), open_side="+x", resolution=resolution),
        _voxel_fragment((1, 0, 0), open_side="-x", resolution=resolution),
    ]
    kwargs = {"resolution": resolution} if weld_units == "voxel" else {}
    merged = merge(fragments, **kwargs).compute()

    assert_watertight(merged)
    assert len(merged.faces) == 20
    np.testing.assert_allclose(merged.volume, 2 * resolution**3)


@pytest.mark.parametrize("merge", [merge_meshes, _merge_mesh_dict_values])
@pytest.mark.parametrize("resolution", [0.01, 0.005])
def test_merge_fragments_weld_tolerates_subvoxel_seam_noise(merge, resolution):
    """The voxel-unit weld closes seams carrying sub-voxel float noise (#20).

    Seam vertices of neighboring chunk fragments may differ by tiny float
    noise well below 0.01 voxel (e.g. from scaling round-trips). An exact
    weld misses such vertices and leaves the seam open; welding on a
    0.01-voxel grid merges them, while distinct vertices (spaced half a
    voxel apart) are still never collapsed.
    """
    fragments = [
        _voxel_fragment((0, 0, 0), open_side="+x", resolution=resolution),
        _voxel_fragment((1, 0, 0), open_side="-x", resolution=resolution),
    ]
    # perturb the seam vertices of the second fragment by 0.004 voxel
    seam = fragments[1].vertices[:, 0] == 1.5 * resolution
    fragments[1].vertices[seam] += np.array([0.004, -0.003, 0.002]) * resolution

    # exact weld: the perturbed seam vertices don't match, seam stays open
    assert not merge(fragments).compute().is_watertight

    # voxel-unit weld: the noise rounds onto the 0.01-voxel grid, seam closes
    merged = merge(fragments, resolution=resolution).compute()
    assert_watertight(merged)
    assert len(merged.faces) == 20
    np.testing.assert_allclose(merged.volume, 2 * resolution**3)
