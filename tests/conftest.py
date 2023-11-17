import pytest


@pytest.fixture(scope="session")
def cebra_project_path():
    """A fixture return a path that conains a valid Cebra project"""

    raise NotImplementedError


@pytest.fixture
def cebra_project(cebra_project_path):
    """A fixture for a valid Cebra project instance"""

    raise NotImplementedError


@pytest.fixture
def cebra_project_with_sources(cebra_project):
    """A fixture for a valid Cebra project instance, incl. added sources"""

    raise NotImplementedError
