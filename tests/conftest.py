import pytest
from .synthetic_data_generator import generate_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_data(seed=42, n_objects=10, object_size=10, object_distance=20):
    """A fixture for a synthetic dataset"""
    project, original_meshes = generate_synthetic_dataset(
        n_objects=n_objects,
        object_size=object_size,
        object_distance=object_distance,
        seed=seed,
    )
    return (project, original_meshes)


@pytest.fixture(scope="session")
def cebra_project_path(synthetic_data):
    """A fixture return a path that conains a valid Cebra project"""
    return synthetic_data[0]


@pytest.fixture(scope="session")
def cebra_project_original_meshes(synthetic_data):
    """A fixture return a path that conains a valid Cebra project"""
    return synthetic_data[1]


@pytest.fixture
def cebra_project(cebra_project_path):
    """A fixture for a valid Cebra project instance"""

    raise NotImplementedError


@pytest.fixture
def cebra_project_with_sources(cebra_project):
    """A fixture for a valid Cebra project instance, incl. added sources"""

    raise NotImplementedError
