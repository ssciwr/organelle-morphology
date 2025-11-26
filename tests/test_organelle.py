import pytest

import numpy as np

from organelle_morphology.organelle import Mitochondrium


def test_organelle_init(project_with_sources):
    p = project_with_sources

    organelles = p.organelles()
    assert len(p.sources["synth_data"].labels) == 19
    assert len(organelles) == 19
    for org in organelles:
        assert isinstance(org, Mitochondrium)
