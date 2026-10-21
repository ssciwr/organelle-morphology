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
    assert posan.own_records[0].meta.dimensionality == 3

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
    res0 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 2, 0.0, 2)
    res1 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 2, 10, 2)
    res2 = posan.density2D(p.sources["synth_data"], (2, 3, 4), 2, 20, 2)
    res3 = posan.density2D(p.sources["synth_data"], (7, 3, 5), 2, 0.0, 1)
    res4 = posan.density2D(p.sources["synth_data"], (7, 3, 5), 2, 10, 1)
    res5 = posan.density2D(p.sources["synth_data"], (7, 3, 5), 2, 30, 1)

    np.testing.assert_almost_equal(res0.mean(), 0.0391694957797291, 5)
    assert res0.shape == (118, 80)
    assert res1.shape == (130, 99)
    assert res2.shape == (138, 116)
    assert res3.shape == (34, 80)
    assert res4.shape == (45, 80)
    assert res5.shape == (62, 80)
    assert res0.ndim == 2

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

    res0 = posan.density1D(p.sources["synth_data"], (2, 3, 4), 0, 0.0, rot_axis=2)
    res1 = posan.density1D(p.sources["synth_data"], (2, 3, 4), 1, 90, rot_axis=2)
    res2 = posan.density1D(p.sources["synth_data"], (2, 3, 4), 2, 90, rot_axis=1)

    np.testing.assert_array_almost_equal(res0, res1)
    np.testing.assert_array_almost_equal(res0, res2)
    assert res0.ndim == 1

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # axes[0, 0].plot(res0)
    # axes[0, 1].plot(res1)
    # axes[0, 2].plot(res2)
    # plt.show()


def test_save_stats(project_with_sources):
    p = project_with_sources
    posan = Position_Analysis(project=p)

    posan.density1D(p.sources["synth_data"], (2, 3, 4), 0, 0.0, 2)
    posan.save_records()
    out_dir = p.path / "analysis" / "PositionProperties"
    assert out_dir.exists()
    assert len(list(out_dir.glob("*.npz"))) == 3
    assert len(list(out_dir.glob("*.yaml"))) == 3


def test_plot(project_with_sources):
    p = project_with_sources
    posan = Position_Analysis(project=p)

    posan.density1D(p.sources["synth_data"], (2, 3, 4), 1, 0.0, 2)

    for rec in posan.own_records:
        if rec.meta.dimensionality == 3:
            ax = posan.plot_density(rec)
        elif rec.meta.dimensionality == 2:
            ax = posan.plot_density(rec)
            assert ax.title._text == "2D Density Plot - mito"
        elif rec.meta.dimensionality == 1:
            ax = posan.plot_density(rec)
            assert ax.title._text == "1D Density Plot - mito"

    fig, axes = posan.plot_multiple_densities(posan.own_records, 3, 1)

    # import matplotlib.pyplot as plt
    # plt.show()


def test_plot_density_3d_axis_order(project_with_sources):
    """3D density plots use x = first array dimension:
    a density nonzero at index (i, j, k) plots at x=i, y=j, z=k."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    posan.density3D(p.sources["synth_data"], (2, 3, 4))
    record = [r for r in posan.own_records if r.meta.dimensionality == 3][0]

    density = np.zeros((7, 9, 11))
    density[2, 3, 5] = 1.0
    record.data.density = density

    ax = posan.plot_density(record)
    x, y, z = ax.collections[0]._offsets3d
    np.testing.assert_array_equal(x, [2])
    np.testing.assert_array_equal(y, [3])
    np.testing.assert_array_equal(z, [5])
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("x", "y", "z")


def test_density3D_anisotropic_axes(project_with_anisotropic_sources):
    """Binning uses per-axis bin sizes aligned with the array axes.
    An x/z axis swap changes shape and object bin positions.
    """
    p = project_with_anisotropic_sources
    s = p.sources["synth_data"]
    posan = Position_Analysis(project=p)

    bin_resolution = (4.0, 4.0, 10.0)  # array order
    density = posan.density3D(s, bin_resolution)

    # array-order resolution (2, 4, 10) gives per-axis bin sizes (2, 1, 1)
    binsize = (2, 1, 1)
    expected_shape = tuple(
        -(-size // b)
        for size, b in zip(s.data.shape, binsize)  # ceil division
    )
    assert density.shape == expected_shape

    # averaging along x yields different numbers than along z
    profile_x = density.sum(axis=(1, 2))
    profile_z = density.sum(axis=(0, 1))
    assert profile_x.shape != profile_z.shape or not np.array_equal(
        profile_x, profile_z
    )

    # the largest organelle must show up in its expected per-axis bin range
    data = np.asarray(s.data)
    labels, counts = np.unique(data, return_counts=True)
    label = int(labels[np.argmax(counts * (labels != 0))])
    coords = np.argwhere(data == label)
    bin_min = coords.min(axis=0) // np.array(binsize)
    bin_max = coords.max(axis=0) // np.array(binsize)
    bin_range = tuple(slice(lo, hi + 1) for lo, hi in zip(bin_min, bin_max))
    assert density[bin_range].max() > 0
