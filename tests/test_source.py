import pytest
from organelle_morphology import source
import numpy as np
from dask.base import compute
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib as mpl


def test_block_mesher_c_in_c(voxels_c_in_c):
    voxels = voxels_c_in_c
    meshes, ids = compute(
        *source._block_mesher(
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
        *source._block_mesher(
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
        *source._block_mesher(
            block=voxels,
            space_offset=(0, 0, 0),
            debug_color=0,
        ),
        scheduler="single-threaded",
    )
    assert len(meshes) == 1
    assert len(ids) == 1


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
        *source._block_mesher(
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
        *source._block_mesher(
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
        *source._block_mesher(
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


def test_curvature(project_with_sources):
    s = list(project_with_sources.sources.values())[0]
    res = s.get_curvature(labels=None, color=True, log=True)
    assert len(res) == 2
    assert len(res[0]) == len(res[1])
    assert all([v.shape == (m.vertices.shape[0],) for v, m in zip(res[0], res[1])])

    res = s.get_curvature(labels=None, color=True, log=False)
    assert len(res) == 2
    assert len(res[0]) == len(res[1])
    assert all([v.shape == (m.vertices.shape[0],) for v, m in zip(res[0], res[1])])

    res = s.get_curvature(labels=None, color=False)
    assert len(res) == 19
    assert all([isinstance(r, np.ndarray) for r in res])

    res = s.get_curvature(labels=s.labels[0], color=False)
    assert len(res) == 1


def test_calculate_mesh(project_with_sources, mocker):
    p = project_with_sources
    s = list(p.sources.values())[0]

    data = np.arange(125).reshape((5, 5, 5))
    data = da.from_array(data)

    mock_data = mocker.patch(
        "organelle_morphology.source.DataSource.data", new_callable=mocker.PropertyMock
    )
    mock_data.return_value = data

    s.calculate_mesh()
    breakpoint()
