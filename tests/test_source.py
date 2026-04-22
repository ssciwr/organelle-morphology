import pytest
from organelle_morphology.block_mesher import block_mesher

import numpy as np
from dask.base import compute
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib as mpl


def test_block_mesher_c_in_c(voxels_c_in_c):
    voxels = voxels_c_in_c
    meshes, ids = compute(
        *block_mesher(
            block=voxels,
            space_offset=(0, 0, 0),
            debug_color=0,
        ),
        scheduler="single-threaded",
    )
    assert len(meshes) == 2
    assert len(ids) == 2


def test_block_mesher_c_through_c(voxels_c_through_c):
    voxels = voxels_c_through_c
    meshes, ids = compute(
        *block_mesher(
            block=voxels,
            space_offset=(0, 0, 0),
            debug_color=0,
        ),
        scheduler="single-threaded",
    )
    assert len(meshes) == 3
    assert len(ids) == 3


def test_block_mesher_c_on_edge(voxels_c_on_edge):
    voxels = voxels_c_on_edge
    meshes, ids = compute(
        *block_mesher(
            block=voxels,
            space_offset=(0, 0, 0),
            debug_color=0,
        ),
        scheduler="single-threaded",
    )
    assert len(meshes) == 1
    assert len(ids) == 1


def test_block_mesher_solid():
    voxels = np.ones((10, 10, 10))
    meshes, ids = compute(
        *block_mesher(
            block=voxels,
            space_offset=(0, 0, 0),
            debug_color=0,
        ),
        scheduler="single-threaded",
    )
    assert len(meshes) == 0
    assert len(ids) == 0


def plot_voxels(voxels):
    """For debugging"""
    cm = mpl.colormaps["tab20"]
    ax = plt.figure().add_subplot(projection="3d")
    for i, label in enumerate(np.unique(voxels)):
        if label == 0:
            continue
        tmp = np.zeros_like(voxels)
        tmp[np.nonzero(voxels == label)] = 1
        ax.voxels(tmp, facecolors=cm.colors[i % 20])

    plt.show(block=False)


@pytest.mark.parametrize("repeat", range(10))
def test_block_mesher_random_no_offset(voxels_random, repeat):
    voxels = voxels_random
    n_labels = np.unique(
        voxels[
            2:29,
            2:29,
            2:29,
        ]
    ).shape[0]
    meshes, ids = compute(
        *block_mesher(
            block=voxels,
            space_offset=(0, 0, 0),
            debug_color=0,
        ),
        scheduler="single-threaded",
    )
    mmesh = sum(meshes.values())
    assert len(meshes) == n_labels - 1
    assert len(ids) == n_labels - 1
    assert mmesh.vertices.min() >= 2.0
    assert mmesh.vertices.max() <= 98.0


@pytest.mark.parametrize("repeat", range(10))
def test_block_mesher_random_offset(voxels_random, repeat):
    voxels = voxels_random
    n_labels = np.unique(
        voxels[
            2:29,
            2:29,
            2:29,
        ]
    ).shape[0]
    meshes, ids = compute(
        *block_mesher(
            block=voxels,
            space_offset=[10, 20, 30],
            debug_color=0,
        )
    )
    mmesh = sum(meshes.values())
    assert len(meshes) == n_labels - 1
    assert len(ids) == n_labels - 1
    assert np.all(
        [mmesh.vertices.min(axis=0)[i] >= a for i, a in enumerate((12, 22, 32))]
    )
    assert np.all(
        [mmesh.vertices.max(axis=0)[i] <= a for i, a in enumerate((38, 48, 58))]
    )


@pytest.mark.parametrize("debug_color", range(4))
def test_block_mesher_debug_colors(voxels_random, debug_color):
    voxels = voxels_random
    n_labels = np.unique(
        voxels[
            2:29,
            2:29,
            2:29,
        ]
    ).shape[0]
    meshes, ids = compute(
        *block_mesher(
            block=voxels,
            space_offset=[10, 20, 30],
            debug_color=debug_color,
        )
    )
    assert len(meshes) == n_labels - 1
    assert len(ids) == n_labels - 1


def test_curvature_map(project_with_sources):
    s = list(project_with_sources.sources.values())[0]
    curvs = s.curvature_map
    assert len(curvs) == 19


@pytest.mark.parametrize("chunksize", range(0, 7))
def test_calculate_mesh(project_with_sources, mocker, chunksize):
    p = project_with_sources
    s = list(p.sources.values())[0]
    if chunksize == 0:
        chunksize = -1

    data = np.arange(343).reshape((7, 7, 7))
    data = da.from_array(data, chunks=chunksize)
    mock_data = mocker.patch("organelle_morphology.source.da.from_array")
    mock_data.return_value = data

    ids_to_chunks, meshes_chunked = s.calculate_mesh()
    assert len(ids_to_chunks) == 342


@pytest.mark.parametrize("chunksize", range(0, 7))
def test_calculate_mesh_clipped(project_with_sources, mocker, chunksize):
    p = project_with_sources
    s = list(p.sources.values())[0]
    if chunksize == 0:
        chunksize = -1

    p.clipping = ((0.3, 0.3, 0.3), (1, 1, 1))

    data = np.arange(1000).reshape((10, 10, 10))
    data = da.from_array(data, chunks=chunksize)
    mock_data = mocker.patch("organelle_morphology.source.da.from_array")
    mock_data.return_value = data

    ids_to_chunks, meshes_chunked = s.calculate_mesh()
    assert len(ids_to_chunks) == 343


@pytest.mark.parametrize("rep", range(20))
def test_calculate_mesh_boarder(project_with_sources, mocker, rep):
    p = project_with_sources
    s = list(p.sources.values())[0]

    o1 = np.random.randint(4) + 1
    o2 = np.random.randint(4) + 1
    o3 = np.random.randint(4) + 1
    data = np.zeros((1000,), dtype=int).reshape((10, 10, 10))
    data[o1 : o1 + 4, o2 : o1 + 4, o3 : o1 + 4] = 1
    data = da.from_array(data, chunks=5)
    mock_data = mocker.patch("organelle_morphology.source.da.from_array")
    mock_data.return_value = data

    # # for debugging:
    # import dask
    #
    # dask.config.set(scheduler="synchronous")
    #

    mesh = list(s.meshes.values())[0].compute()
    assert mesh.is_watertight
    assert len(list(s.meshes.keys())) == 1
    assert (
        np.count_nonzero(np.unique(mesh.vertices, axis=0, return_counts=True)[1] != 1)
        == 0
    )


def test_get_meshes_curvature_colored(project_with_sources, mocker):
    p = project_with_sources
    s = list(p.sources.values())[0]

    mock_calc_curv = mocker.patch.object(s, "calc_curvature")
    s._curvature_map = {
        la: np.ones(s.meshes[la].compute().vertices.shape[0]) for la in s.labels
    }

    meshes = s.get_meshes_curvature_colored(labels=None)
    assert len(meshes) == 19
    mock_calc_curv.assert_called_with(s.labels)
    meshes = s.get_meshes_curvature_colored(labels=1)
    assert len(meshes) == 1
    meshes = s.get_meshes_curvature_colored(labels=[3, 2])
    assert np.all(sum(meshes).compute().visual.vertex_colors == [5, 48, 97, 255])
    assert len(meshes) == 2


def test_calc_curvature(project_with_sources, mocker):
    """Test the calc_curvature method directly"""
    s = list(project_with_sources.sources.values())[0]
    s.labels  # compute before mocking `compute`

    mock_logger = mocker.patch.object(s, "logger")
    mock_compute = mocker.patch("organelle_morphology.source.compute")

    mock_compute.return_value = s.labels  # list as long as the labels
    ret = s.calc_curvature(labels=None)

    assert ret is s._curvature_map
    assert len(s._curvature_map) == 19
    mock_compute.assert_called()

    mock_compute.reset_mock()
    mock_logger.debug.reset_mock()

    s.calc_curvature(labels=s.labels[5])

    mock_compute.assert_not_called()
    mock_logger.debug.assert_called_with("All curvatures already calculated.")

    mock_logger.debug.reset_mock()
    labels_to_clear = s.labels[:3]
    for label in labels_to_clear:
        if label in s._curvature_map:
            del s._curvature_map[label]

    s.calc_curvature(labels=labels_to_clear + [s.labels[5]])

    assert len(s._curvature_map) == 19


def test_mcs_dicts(project_with_sources):
    p = project_with_sources
    s = list(project_with_sources.sources.values())[0]

    if len(ps := list(p.path.glob("cache*"))):
        raise RuntimeError(f"Existing caches found!!\n{ps}")
    p.search_mcs(10)
    mcs_dicts = s.mcs_dicts
    assert list(mcs_dicts.keys()) == ["0.0-10,-"]
    assert len(mcs_dicts["0.0-10,-"]) == 10
    assert len(mcs_dicts["0.0-10,-"]["mito_0019"]) == 2
    assert all(
        k in mcs_dicts["0.0-10,-"]["mito_0019"]["mito_0015"].keys()
        for k in ["area", "distances", "vertices_index"]
    )


def test_get_centroid(project_with_sources):
    p = project_with_sources
    center = p.sources["synth_data"].centroid
    np.testing.assert_allclose(center, [126.11414925, 138.18867658, 186.33403298])


def test_get_coarse_centroid(project_with_sources):
    p = project_with_sources
    center = p.sources["synth_data"].global_coarse_centroid
    np.testing.assert_allclose(center, [120.444444, 129.666667, 180.666667])
