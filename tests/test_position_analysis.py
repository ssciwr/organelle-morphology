import dataclasses
import logging

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
    """3D density plots use x = first array dimension, in micrometers:
    a density nonzero at index (i, j, k) plots at x=i*bin_width[0], etc."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    posan.density3D(p.sources["synth_data"], (2, 3, 4))
    record = [r for r in posan.own_records if r.meta.dimensionality == 3][0]

    density = np.zeros((7, 9, 11))
    density[2, 3, 5] = 1.0
    record.data.density = density

    ax = posan.plot_density(record)
    x, y, z = ax.collections[0]._offsets3d
    # isotropic 1 um voxels: bin widths (2, 3, 4) um
    np.testing.assert_array_equal(x, [4])
    np.testing.assert_array_equal(y, [9])
    np.testing.assert_array_equal(z, [20])
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == (
        "x [$\\mu$m]",
        "y [$\\mu$m]",
        "z [$\\mu$m]",
    )
    # the full binned volume is visible, not only the filled bins
    np.testing.assert_allclose(
        (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()),
        ((0, 14), (0, 27), (0, 44)),
    )


def test_plot_density_2d_axes_and_units(project_with_sources):
    """2D density plots put the first array axis on the horizontal plot
    axis and the second on the vertical one, both in micrometers."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    posan.density2D(p.sources["synth_data"], (2, 3, 4), marginal_axis=2)
    record = [r for r in posan.own_records if r.meta.dimensionality == 2][0]

    density = np.zeros((5, 7))
    density[2, 4] = 1.0
    record.data.density = density

    ax = posan.plot_density(record)
    img = ax.images[0]
    # transposed: density axis 0 -> horizontal, axis 1 -> vertical
    np.testing.assert_array_equal(img.get_array(), density.T)
    # marginal axis 2 (z) dropped: remaining bin widths (2, 3) um
    np.testing.assert_allclose(img.get_extent(), (0, 5 * 2, 0, 7 * 3))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x [$\\mu$m]", "y [$\\mu$m]")


def test_plot_density_1d_units(project_with_sources):
    """1D density plots measure along the selected array axis, in
    micrometers."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    posan.density1D(p.sources["synth_data"], (2, 3, 4), axis=1)
    record = [r for r in posan.own_records if r.meta.dimensionality == 1][0]

    density = np.array([0.0, 1.0, 0.5, 0.0])
    record.data.density = density

    ax = posan.plot_density(record)
    # axis 1 (y): bin width 3 um
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0, 3, 6, 9])
    assert ax.get_xlabel() == "y [$\\mu$m]"


def test_plot_density_3d_clipping_offset(project_with_sources):
    """With clipping, 3D density plot positions are offset by the lower
    clipping corner of the source: they describe the position within the
    full data extent, like the show scene."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    p.clipping = ((0.5, 0.25, 0.0), (1.0, 1.0, 1.0))
    posan.density3D(p.sources["synth_data"], (2, 3, 4))
    record = next(r for r in posan.own_records if r.meta.dimensionality == 3)

    density = np.zeros((7, 9, 11))
    density[2, 3, 5] = 1.0
    record.data.density = density

    corner = p.sources["synth_data"].clipping_corners[0]
    # x and y are actually clipped, so the offset is real
    assert corner[0] > 0 and corner[1] > 0
    # isotropic 1 um voxels: bin widths (2, 3, 4) um
    widths = record.meta.bin_width
    assert widths == (2.0, 3.0, 4.0)

    ax = posan.plot_density(record)
    x, y, z = ax.collections[0]._offsets3d
    np.testing.assert_allclose(x, [corner[0] + 2 * widths[0]])
    np.testing.assert_allclose(y, [corner[1] + 3 * widths[1]])
    np.testing.assert_allclose(z, [corner[2] + 5 * widths[2]])
    # the full binned volume is visible, offset by the corner
    np.testing.assert_allclose(
        (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()),
        (
            (corner[0], corner[0] + 7 * widths[0]),
            (corner[1], corner[1] + 9 * widths[1]),
            (corner[2], corner[2] + 11 * widths[2]),
        ),
    )


def test_plot_density_2d_clipping_offset(project_with_sources):
    """With clipping, the 2D density plot extent is offset by the lower
    clipping corner of the source."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    p.clipping = ((0.5, 0.25, 0.0), (1.0, 1.0, 1.0))
    posan.density2D(p.sources["synth_data"], (2, 3, 4), marginal_axis=2)
    record = next(r for r in posan.own_records if r.meta.dimensionality == 2)

    density = np.zeros((5, 7))
    density[2, 4] = 1.0
    record.data.density = density

    corner = p.sources["synth_data"].clipping_corners[0]
    ax = posan.plot_density(record)
    img = ax.images[0]
    # marginal axis 2 (z) dropped from widths and offset: (x, y) remain
    np.testing.assert_allclose(
        img.get_extent(),
        (corner[0], corner[0] + 5 * 2, corner[1], corner[1] + 7 * 3),
    )


def test_plot_density_1d_clipping_offset(project_with_sources):
    """With clipping, the 1D density plot x axis is offset by the lower
    clipping corner of the source along the measured axis."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    p.clipping = ((0.5, 0.25, 0.0), (1.0, 1.0, 1.0))
    posan.density1D(p.sources["synth_data"], (2, 3, 4), axis=1)
    record = next(r for r in posan.own_records if r.meta.dimensionality == 1)

    density = np.array([0.0, 1.0, 0.5, 0.0])
    record.data.density = density

    corner = p.sources["synth_data"].clipping_corners[0]
    # y is actually clipped, so the offset is real
    assert corner[1] > 0

    ax = posan.plot_density(record)
    # axis 1 (y): bin width 3 um, offset by the lower clipping corner
    np.testing.assert_allclose(
        ax.lines[0].get_xdata(),
        corner[1] + np.arange(4) * record.meta.bin_width[1],
    )
    assert ax.get_xlabel() == "y [$\\mu$m]"


def test_plot_density_unknown_source_zero_offset(project_with_sources, caplog):
    """A record whose source is not loaded in the project plots without a
    clipping offset instead of crashing, with a warning."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    posan.density3D(p.sources["synth_data"], (2, 3, 4))
    record = next(r for r in posan.own_records if r.meta.dimensionality == 3)

    record.meta = dataclasses.replace(record.meta, source="not_loaded")
    density = np.zeros((7, 9, 11))
    density[2, 3, 5] = 1.0
    record.data.density = density

    with caplog.at_level(logging.WARNING, logger="organelle_morphology.position"):
        ax = posan.plot_density(record)
    assert "not_loaded" in caplog.text
    x, y, z = ax.collections[0]._offsets3d
    # zero offset: isotropic 1 um voxels, bin widths (2, 3, 4) um
    np.testing.assert_array_equal(x, [4])
    np.testing.assert_array_equal(y, [9])
    np.testing.assert_array_equal(z, [20])


def test_plot_density_legacy_record_without_bin_width(project_with_sources):
    """Records saved before bin_width existed fall back to the requested
    bin_resolution for their plot axes."""
    p = project_with_sources
    posan = Position_Analysis(project=p)
    posan.density3D(p.sources["synth_data"], (2, 3, 4))
    record = [r for r in posan.own_records if r.meta.dimensionality == 3][0]

    record.meta = dataclasses.replace(record.meta, bin_width=None)
    density = np.zeros((7, 9, 11))
    density[1, 2, 3] = 1.0
    record.data.density = density

    ax = posan.plot_density(record)
    x, y, z = ax.collections[0]._offsets3d
    np.testing.assert_array_equal(x, [2])
    np.testing.assert_array_equal(y, [6])
    np.testing.assert_array_equal(z, [12])


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


def test_density3D_bin_width_meta(project_with_anisotropic_sources):
    """The record meta stores the actual per-axis bin width in
    micrometers (voxels per bin * voxel resolution), not the requested
    bin resolution."""
    p = project_with_anisotropic_sources
    s = p.sources["synth_data"]
    posan = Position_Analysis(project=p)

    posan.density3D(s, (4.0, 4.0, 10.0))
    record = [r for r in posan.own_records if r.meta.dimensionality == 3][0]

    # array-order resolution (2, 4, 10) um/voxel, binsize (2, 1, 1) voxels
    assert record.meta.bin_width == (4.0, 4.0, 10.0)


def test_density3D_binsize_below_voxel_size(project_with_anisotropic_sources):
    """A bin resolution below the voxel resolution falls back to one
    voxel per bin instead of crashing."""
    p = project_with_anisotropic_sources
    s = p.sources["synth_data"]
    posan = Position_Analysis(project=p)

    # voxel resolution at s0 is (2, 4, 10) um, the request is smaller
    density = posan.density3D(s, (1.0, 1.0, 5.0))
    assert density.shape == tuple(s.data.shape)

    record = [r for r in posan.own_records if r.meta.dimensionality == 3][0]
    assert record.meta.bin_width == (2.0, 4.0, 10.0)
