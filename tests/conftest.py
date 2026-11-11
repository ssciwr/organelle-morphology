from functools import reduce
from pathlib import Path
from typing import Optional


from distributed import LocalCluster
from distributed.utils_test import (
    client,  # noqa: F401
    loop,  # noqa: F401
    cluster_fixture,  # noqa: F401
    loop_in_thread,  # noqa: F401
    cleanup,  # noqa: F401
)
from organelle_morphology.project import Project

import numpy as np
import pytest
import shutil
from .synthetic_data_generator import generate_synthetic_dataset

try:
    import resource
except ImportError:
    resource = None


def cubify(p, x, y, z, min_s: int = 3, max_s: Optional[int] = 6, z_offset=0):
    """Turn point to cube"""
    if max_s is not None:
        assert min_s < max_s
        cube = (
            ((p[0] + np.random.randint(min_s, max_s)) > x)
            & (x >= (p[0] - np.random.randint(min_s, max_s)))
            & ((p[1] + np.random.randint(min_s, max_s)) > y)
            & (y >= (p[1] - np.random.randint(min_s, max_s)))
            & ((p[2] + np.random.randint(min_s, max_s)) > z)
            & (z >= (p[2] - np.random.randint(min_s, max_s)))
        )
    else:
        cube = (
            ((p[0] + min_s) > x)
            & (x >= (p[0] - min_s))
            & ((p[1] + min_s) > y)
            & (y >= (p[1] - min_s))
            & ((p[2] + min_s + z_offset) > z)
            & (z >= (p[2] - min_s - z_offset))
        )

    return cube


@pytest.fixture
def voxels_c_in_c(size=100):
    cubes = []
    x, y, z = np.indices((size, size, size))

    # cube in cube
    cube = cubify([50, 50, 50], x, y, z, 3, None, 0)
    n_cube = np.zeros_like(cube, dtype=float)
    n_cube[cube] = 1
    cubes.append(n_cube)
    cube = cubify([50, 50, 50], x, y, z, 6, None, 0)
    n_cube = np.zeros_like(cube, dtype=float)
    n_cube[cube] = 2
    cubes.append(n_cube)

    return reduce(np.add, cubes)


@pytest.fixture
def voxels_c_through_c(size=100):
    cubes = []
    x, y, z = np.indices((size, size, size))

    # cube split by cube
    cube = cubify([35, 50, 50], x, y, z, 6, None, -4)
    n_cube = np.zeros_like(cube, dtype=float)
    n_cube[cube] = 2
    cubes.append(n_cube)
    cube = cubify([35, 50, 50], x, y, z, 3, None, 2)
    n_cube = np.zeros_like(cube, dtype=int)
    n_cube[cube] = 3
    cubes.append(n_cube)

    return reduce(np.add, cubes)


@pytest.fixture
def voxels_c_on_edge(size=100):
    cubes = []
    x, y, z = np.indices((size, size, size))

    static_points = np.array(
        [
            [0, 0, 0],  # completely outside
            [20, 2, 1],  # partially outside
        ]
    )
    for i, p in enumerate(static_points):
        cube = cubify(p, x, y, z, 2, None)
        n_cube = np.zeros_like(cube, dtype=int)
        n_cube[cube] = i + 1
        cubes.append(n_cube)

    return reduce(np.add, cubes)


@pytest.fixture
def voxels_random(size=30, n_points=5):
    points = np.random.randint(0, size, n_points * 3).reshape((-1, 3))
    x, y, z = np.indices((size, size, size))

    cubes = []

    for i, p in enumerate(points):
        cube = cubify(p, x, y, z)
        n_cube = np.zeros_like(cube, dtype=int)
        n_cube[cube] = i + 1
        cubes.append(n_cube)

    return reduce(np.add, cubes)


@pytest.fixture(scope="session")
def _synthetic_data(n_objects=30, object_size=20, object_distance=100, seed=42):
    """A fixture for a synthetic dataset"""
    project_path, original_meshes = generate_synthetic_dataset(
        n_objects=n_objects,
        object_size=object_size,
        object_distance=object_distance,
        seed=seed,
    )
    return (project_path, original_meshes)


@pytest.fixture(scope="session")
def _synthetic_data_fine_resolution():
    """A fixture for a synthetic dataset at 0.01 um/voxel.

    Regression fixture for #20: at fine physical resolutions the mesh
    vertices (spaced by half a voxel) sit closer together than a fixed
    0.01 um weld grid, so welding chunk fragments must not use one.
    """
    return generate_synthetic_dataset(
        n_objects=5,
        object_size=20,
        object_distance=100,
        seed=43,
        resolution=(0.01, 0.01, 0.01),
    )


def _symlink_dataset(project_path: Path, tmp_path: Path) -> Path:
    """Create a symlinked copy of a dataset in a separate dir.

    Necessary so projects don't share caches.
    """
    new_project = tmp_path / project_path.name
    new_project.mkdir()
    for item in project_path.iterdir():
        dest = new_project / item.name
        try:
            dest.symlink_to(item)
        except OSError:
            # Fallback for Windows users without Admin rights
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    return new_project


@pytest.fixture(scope="session")
def _anisotropic_synthetic_data(
    n_objects=30, object_size=20, object_distance=100, seed=42
):
    """A fixture for an anisotropic synthetic dataset.

    Resolution and downsampling factors are non-palindromic and
    axis-asymmetric (given in disk order), so that any axis swap changes
    the numerical results.
    """
    project_path, original_meshes = generate_synthetic_dataset(
        n_objects=n_objects,
        object_size=object_size,
        object_distance=object_distance,
        seed=seed,
        resolution=[10, 4, 2],
        downsampling_factors=([1, 1, 1], [2, 2, 2], [4, 8, 2], [16, 16, 16]),
    )
    return (project_path, original_meshes)


@pytest.fixture(scope="function")
def synthetic_data(_synthetic_data, tmp_path: Path):
    """Creating symlink in separate dir for each function call
    Necessary so projects don't share caches.
    """
    project_path, original_meshes = _synthetic_data
    return (_symlink_dataset(project_path, tmp_path), original_meshes)


@pytest.fixture(scope="function")
def anisotropic_synthetic_data(_anisotropic_synthetic_data, tmp_path: Path):
    """Function-scoped working copy of the anisotropic dataset."""
    _project_path, original_meshes = _anisotropic_synthetic_data
    return (_symlink_dataset(_project_path, tmp_path), original_meshes)


@pytest.fixture()
def project_path(synthetic_data):
    """Returns a path that conains a valid project"""
    return synthetic_data[0]


@pytest.fixture()
def anisotropic_project_path(anisotropic_synthetic_data):
    """Returns a path that contains an anisotropic project"""
    return anisotropic_synthetic_data[0]


@pytest.fixture(scope="session")
def cluster():
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_NOFILE, (10000, 10000))

    cluster = LocalCluster(dashboard_address=None)
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def custom_client(cluster):
    cclient = cluster.get_client()
    yield cclient
    cclient.close()


@pytest.fixture
def project(project_path, client):  # noqa: F811
    """A fixture for a valid project instance"""
    project = Project(project_path, client=client)
    yield project
    project.clear_caches(True, silent=True)


@pytest.fixture
def anisotropic_project(anisotropic_project_path, client):  # noqa: F811
    """A fixture for a valid project instance on anisotropic data"""
    project = Project(anisotropic_project_path, client=client)
    yield project
    project.clear_caches(True, silent=True)


@pytest.fixture
def project_with_sources(project):
    """A fixture for a valid project instance, incl. added sources"""
    project.add_source("synth_data", "mito")
    yield project
    project.clear_caches(True, silent=True)


@pytest.fixture
def project_with_sources_fine_resolution(
    _synthetic_data_fine_resolution,
    tmp_path: Path,
    client,  # noqa: F811
):
    """A project with sources at 0.01 um/voxel (regression fixture for #20)"""
    project_path, _ = _synthetic_data_fine_resolution
    project = Project(_symlink_dataset(project_path, tmp_path), client=client)
    project.add_source("synth_data", "mito")
    yield project
    project.clear_caches(True, silent=True)


@pytest.fixture
def project_with_anisotropic_sources(anisotropic_project):
    """A fixture for an anisotropic project instance, incl. added sources"""
    anisotropic_project.add_source("synth_data", "mito")
    yield anisotropic_project
    anisotropic_project.clear_caches(True, silent=True)
