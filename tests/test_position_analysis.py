from organelle_morphology.position import Position_Analysis
import organelle_morphology.position

import numpy as np


def test_density3D(project_with_sources):
    p = project_with_sources
    posan = Position_Analysis(project=p)
    density = posan.density3D(p.sources["synth_data"], (2, 3, 4))
    assert density.shape == (118, 80, 83)


def test_density3D_cached(project_with_sources, mocker):
    p = project_with_sources
    mock_coarsen = mocker.spy(organelle_morphology.position.da, "coarsen")
    posan = Position_Analysis(project=p)
    density = posan.density3D(p.sources["synth_data"], (2, 3, 4))
    mock_coarsen.assert_called_once()
    assert density.shape == (118, 80, 83)
    density_cached = posan.density3D(p.sources["synth_data"], (2, 3, 4))
    mock_coarsen.assert_called_once()
    np.testing.assert_array_almost_equal(density_cached, density)
    p.clear_caches()
    density_cached = posan.density3D(p.sources["synth_data"], (2, 3, 4))
    mock_coarsen.assert_called_once()
    np.testing.assert_array_almost_equal(density_cached, density)


def test_density2D(project_with_sources):
    p = project_with_sources
    posan = Position_Analysis(project=p)
    res0 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 2, 0.0, (0, 1))
    res1 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 2, 10, (0, 1))
    res2 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 2, 20, (0, 1))
    res3 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 1, 0.0, (0, 1))
    res4 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 1, 10, (2, 1))
    res5 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 1, 20, (2, 1))

    np.testing.assert_almost_equal(res0.mean(), 0.0391694957797291, 5)
    assert res0.shape == (118, 80)
    assert res1.shape == (130, 99)
    assert res2.shape == (138, 116)
    assert res3.shape == (118, 83)
    assert res4.shape == (118, 96)
    assert res5.shape == (118, 105)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # axes[0, 0].imshow(res0)
    # axes[0, 1].imshow(res1)
    # axes[0, 2].imshow(res2)
    # axes[1, 0].imshow(res3)
    # axes[1, 1].imshow(res4)
    # axes[1, 2].imshow(res5)
    # plt.show()


def test_density1D(project_with_sources):
    p = project_with_sources
    posan = Position_Analysis(project=p)

    res0 = posan.density1D(p.sources["synth_data"], (2, 3, 4), 0, 0.0, (0, 1))
    res1 = posan.density1D(p.sources["synth_data"], (2, 3, 4), 1, 90, (0, 1))
    res2 = posan.density1D(p.sources["synth_data"], (2, 3, 4), 2, 90, (0, 2))

    np.testing.assert_array_almost_equal(res0, res1)
    np.testing.assert_array_almost_equal(res0, res2)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # axes[0, 0].plot(res0)
    # axes[0, 1].plot(res1)
    # axes[0, 2].plot(res2)
    # plt.show()
