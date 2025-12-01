import pytest

import numpy as np

from organelle_morphology.organelle import Mitochondrium


def test_organelle_init(project_with_sources):
    p = project_with_sources

    organelles = p.organelles
    assert len(p.sources["synth_data"].labels) == 19
    assert len(organelles) == 19
    for org in organelles:
        assert isinstance(org, Mitochondrium)


def test_organelle_curvature(project_with_sources):
    for org in project_with_sources.organelles:
        curv = org.curvature_map
        assert isinstance(curv, np.ndarray)
        assert curv.shape == (org.mesh.compute().vertices.shape[0],)
