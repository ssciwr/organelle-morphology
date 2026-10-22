import pytest
from organelle_morphology.block_mesher import block_mesher

import xml.etree.ElementTree as ET

import numpy as np
from dask.base import compute
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib as mpl

from .mesh_asserts import assert_watertight
from organelle_morphology.project import Project

from .synthetic_data_generator import generate_synthetic_dataset


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
        # scheduler="single-threaded",
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


def test_calculate_mesh(project_with_sources, mocker):
    p = project_with_sources
    s = list(p.sources.values())[0]
    for chunksize in range(0, 7):
        if chunksize == 0:
            chunksize = -1

        data = np.arange(343).reshape((7, 7, 7))
        data = da.from_array(data, chunks=chunksize)
        mock_data = mocker.patch("organelle_morphology.source.da.from_array")
        mock_data.return_value = data

        ids_to_chunks, meshes_chunked = s.calculate_mesh()
        assert len(ids_to_chunks) == 342


def test_calculate_mesh_clipped(project_with_sources, mocker):
    p = project_with_sources
    s = list(p.sources.values())[0]
    for chunksize in range(0, 7):
        if chunksize == 0:
            chunksize = -1

        p.clipping = ((0.3, 0.3, 0.3), (1, 1, 1))

        data = np.arange(1000).reshape((10, 10, 10))
        data = da.from_array(data, chunks=chunksize)
        mock_data = mocker.patch("organelle_morphology.source.da.from_array")
        mock_data.return_value = data

        ids_to_chunks, meshes_chunked = s.calculate_mesh()
        assert len(ids_to_chunks) == 343


@pytest.mark.parametrize(
    ("project_fixture", "resolution"),
    [
        ("project_with_sources", 1.0),
        ("project_with_sources_fine_resolution", 0.01),
    ],
)
def test_calculate_mesh_boarder(request, mocker, project_fixture, resolution):
    """Meshes of labels crossing chunk borders must be watertight.

    Parametrized over the physical resolution (regression test for #20):
    at 0.01 um/voxel the mesh vertex spacing (half a voxel) is finer than
    the fixed 0.01 um weld grid that used to merge the chunk fragments,
    which collapsed vertices and broke watertightness.
    """
    p = request.getfixturevalue(project_fixture)
    p.simplify = 0.0
    s = list(p.sources.values())[0]
    assert s.resolution == (resolution, resolution, resolution)

    # cube placements close to the data borders; o >= 2 puts the 4-voxel cube
    # across the chunk borders at index 5 of the 10x10x10 test data
    offsets = [(o, o, o) for o in (1, 2, 3, 4)]
    offsets += [(2, 4, 3), (4, 3, 2), (3, 2, 4), (4, 4, 4)]

    saw_multi_chunk = False
    for o1, o2, o3 in offsets:
        # stopall (not resetall): the patched `da.from_array` is also used
        # below to build the data and must be the real one again
        mocker.stopall()
        p.clear_caches(True, silent=True)
        data = np.zeros((1000,), dtype=int).reshape((10, 10, 10))
        data[o1 : o1 + 4, o2 : o2 + 4, o3 : o3 + 4] = 1
        data = da.from_array(data, chunks=5)
        mock_data = mocker.patch("organelle_morphology.source.da.from_array")
        mock_data.return_value = data

        # # for debugging:
        # import dask
        #
        # dask.config.set(scheduler="synchronous")
        #

        mesh = list(s.meshes.values())[0].compute()
        assert_watertight(mesh)
        assert len(list(s.meshes.keys())) == 1
        assert (
            np.count_nonzero(
                np.unique(mesh.vertices, axis=0, return_counts=True)[1] != 1
            )
            == 0
        )
        if any(len(chunks) > 1 for chunks in s.ids_to_chunks.values()):
            saw_multi_chunk = True

    assert saw_multi_chunk, "cube placements never crossed a chunk border"


def test_meshes_stay_watertight_with_simplify(project_with_sources, mocker):
    p = project_with_sources
    s = list(p.sources.values())[0]

    face_counts = {}
    for simplify in (0.0, 0.5):
        p.simplify = simplify
        for _ in range(3):
            mocker.resetall()
            p.clear_memory_cache()
            o1 = np.random.randint(2) + 1
            o2 = np.random.randint(2) + 1
            o3 = np.random.randint(2) + 1
            data = np.zeros((1000,), dtype=int).reshape((10, 10, 10))
            x, y, z = np.indices((7, 7, 7))
            ball = ((x - 3) ** 2 + (y - 3) ** 2 + (z - 3) ** 2) <= 2.9**2
            data[o1 : o1 + 7, o2 : o2 + 7, o3 : o3 + 7] = ball
            data = da.from_array(data, chunks=5)
            mock_data = mocker.patch("organelle_morphology.source.da.from_array")
            mock_data.return_value = data

            mesh = list(s.meshes.values())[0].compute()
            assert mesh.is_watertight
            assert mesh.is_winding_consistent
            assert mesh.euler_number == 2
            faces = np.asarray(mesh.faces)
            assert not (
                (faces[:, 0] == faces[:, 1])
                | (faces[:, 1] == faces[:, 2])
                | (faces[:, 0] == faces[:, 2])
            ).any()
            face_counts.setdefault(simplify, []).append(len(faces))

    # simplification actually decimated the meshes
    assert max(face_counts[0.5]) < min(face_counts[0.0])


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
    assert len(mcs_dicts["0.0-10,-"]) == 11
    assert len(mcs_dicts["0.0-10,-"]["mito_0019"]) == 3
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


def test_anisotropic_metadata_in_array_order(project_with_anisotropic_sources):
    """All metadata tuples are in array order.

    The anisotropic fixture has disk-order voxel size [10, 4, 2] and a
    disk-order downsampling factor (4, 8, 2) at level s2; converted to
    array order these become (2, 4, 10) and (2, 8, 4). Any axis swap
    changes the expected values.
    """
    p = project_with_anisotropic_sources
    s = p.sources["synth_data"]

    assert s.metadata.size == s.data.shape
    assert s.data_resolution == (2.0, 4.0, 10.0)
    assert s.metadata.downsampling[2] == (2, 8, 4)

    # per array axis: resolution = data resolution * downsampling factor
    p.compression_level = "s2"
    assert s.resolution == (2.0 * 2, 4.0 * 8, 10.0 * 4)


def test_metadata_resolution_mismatch_raises(tmp_path, client):
    """Metadata load fails loudly when XML voxelSize and n5 resolution
    disagree (both are compared in disk order)."""
    dataset_dir, _ = generate_synthetic_dataset(
        n_objects=2,
        object_size=15,
        object_distance=60,
        seed=7,
        working_dir=tmp_path,
        resolution=[6, 3, 1],
    )

    # tamper with the XML voxel size so it no longer matches the n5 attrs
    xml_path = dataset_dir / "synth_data.xml"
    tree = ET.parse(xml_path)
    voxel_size = (
        tree.getroot()
        .find("SequenceDescription")
        .find("ViewSetups")
        .find("ViewSetup")
        .find("voxelSize")
        .find("size")
    )
    voxel_size.text = "9 9 9"
    tree.write(xml_path)

    project = Project(dataset_dir, client=client)
    with pytest.raises(AssertionError, match="resolution disagree"):
        project.add_source("synth_data", "mito")


def test_anisotropic_regionprops_spacing(project_with_anisotropic_sources):
    """Regionprops uses per-axis spacing aligned with the array axes:
    voxel volume = voxel count * product of the per-axis spacing and
    centroid_i = mean index_i * spacing_i."""
    p = project_with_anisotropic_sources
    s = p.sources["synth_data"]
    p.compression_level = "s2"

    spacing = np.array(s.resolution)  # array order, non-palindromic
    data = np.asarray(s.data)

    labels, counts = np.unique(data, return_counts=True)
    label = int(labels[np.argmax(counts * (labels != 0))])
    props = s.basic_geometric_properties[f"mito_{label:04d}"]

    coords = np.argwhere(data == label)
    assert props["voxel_volume"] == pytest.approx(len(coords) * np.prod(spacing))
    np.testing.assert_allclose(props["voxel_centroid"], coords.mean(axis=0) * spacing)
