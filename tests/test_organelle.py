import numpy as np
from tqdm import tqdm

from organelle_morphology.organelle import Mitochondrium


def test_organelle_init(project_with_sources):
    p = project_with_sources

    organelles = p.organelles
    assert len(p.sources["synth_data"].labels) == 19
    assert len(organelles) == 19
    for org in organelles:
        assert isinstance(org, Mitochondrium)


def test_organelle_curvature(project_with_sources, mocker):
    s = project_with_sources.sources["synth_data"]
    mock_calc = mocker.patch.object(s, "calc_curvature")
    for org in tqdm(project_with_sources.organelles):
        mock_calc.return_value = {org.label: np.ones((5,))}
        curv = org.curvature_map
        assert isinstance(curv, np.ndarray)
        mock_calc.assert_called_with(org.label)
        mock_calc.reset_mock()
