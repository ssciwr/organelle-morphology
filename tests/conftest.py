from functools import reduce
from pathlib import Path
from typing import Optional


from dask.distributed import LocalCluster
from distributed.utils_test import (
    client,  # noqa: F401
    loop,  # noqa: F401
    cluster_fixture,  # noqa: F401
    loop_in_thread,  # noqa: F401
    cleanup,  # noqa: F401
)
from organelle_morphology import Project

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


@pytest.fixture(scope="function")
def synthetic_data(_synthetic_data, tmp_path: Path):
    """Creating symlink in separate dir for each function call
    Necessary so projects don't share caches.
    """
    project_path, original_meshes = _synthetic_data
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

    return (new_project, original_meshes)


@pytest.fixture()
def project_path(synthetic_data):
    """Returns a path that conains a valid project"""
    return synthetic_data[0]


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
def project(project_path, custom_client):
    """A fixture for a valid project instance"""
    project = Project(project_path, client=custom_client)
    yield project
    project.clear_caches(True)


@pytest.fixture
def project_with_sources(project):
    """A fixture for a valid project instance, incl. added sources"""
    project.add_source("synth_data", "mito")
    yield project
    project.clear_caches(True)
