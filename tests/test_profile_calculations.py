import numpy as np
from organelle_morphology.profile_calculations import (
    ProfileCalculator,
    ProfileProperties,
    ProfileMeta,
)
from organelle_morphology.statistics import Stats


def test_calculate_profile_lengths(project_with_sources):
    """Regression-Test the 2D profile length calculation pipeline."""
    calculator = ProfileCalculator(project_with_sources)

    # Calculation returns None and updates project_with_sources.stats
    calculator.calculate_profile_lengths(ids="mito_0007", axis="z", num_slices=3)

    # Retrieve the stat from the central project stats object
    for s in project_with_sources.stats:
        if isinstance(s.data, ProfileProperties) and s.meta.organelle_id == "mito_0007":
            profile_stat = s
            break
    else:
        raise ValueError("Profile stat for mito_0007 not found in project stats")

    assert isinstance(profile_stat, Stats)
    assert isinstance(profile_stat.data, ProfileProperties)
    assert isinstance(profile_stat.meta, ProfileMeta)

    assert profile_stat.meta.organelle_id == "mito_0007"
    assert profile_stat.meta.axis_used == "z"
    assert profile_stat.meta.num_slices_attempted == 3

    expected_perimeters = [33.53122292, 39.50609665, 29.67766953]
    assert len(profile_stat.data.perimeters) == len(expected_perimeters)
    np.testing.assert_almost_equal(
        profile_stat.data.perimeters, expected_perimeters, decimal=5
    )
    np.testing.assert_almost_equal(
        profile_stat.data.mean_perimeter, np.mean(expected_perimeters), decimal=5
    )

    expected_widths = [11.096170510586074, 14.422205101855956, 9.848857801796104]
    assert len(profile_stat.data.widths) == len(expected_widths)
    np.testing.assert_almost_equal(profile_stat.data.widths, expected_widths, decimal=5)
    np.testing.assert_almost_equal(
        profile_stat.data.mean_width, np.mean(expected_widths), decimal=5
    )

    expected_ratios = [0.33092054342827526, 0.36506277064064424, 0.3318608892774375]
    assert len(profile_stat.data.ratios) == len(expected_ratios)
    np.testing.assert_almost_equal(profile_stat.data.ratios, expected_ratios, decimal=5)
    np.testing.assert_almost_equal(
        profile_stat.data.mean_ratio, np.mean(expected_ratios), decimal=5
    )

    print(profile_stat.data.widths)
    print(profile_stat.data.ratios)


def test_calculate_random_profiles(project_with_sources):
    """Regression-Test the 2D profile length with random planes."""
    calculator = ProfileCalculator(project_with_sources)
    calculator.calculate_random_profiles(ids="mito_0007", num_planes=5, seed=42)

    # Retrieve from the central registry
    for s in project_with_sources.stats:
        if isinstance(s.data, ProfileProperties) and s.meta.organelle_id == "mito_0007":
            profile_stat = s
            break
    else:
        raise ValueError("Profile stat for mito_0007 not found in project stats")

    perimeters = profile_stat.data.perimeters
    expected = [31.36676084, 52.20270082, 32.24204422, 22.81741491, 30.23820865]

    np.testing.assert_almost_equal(perimeters, expected, decimal=5)


def test_calculate_skeleton_profiles(project_with_sources):
    """Regression-Test the 2D profile length with skeleton profiles."""
    # Generate the skeleton first
    project_with_sources.skeletonize_wavefront(ids="mito_0007")

    calculator = ProfileCalculator(project_with_sources)
    calculator.calculate_skeleton_profiles(ids="mito_0007")

    # Retrieve from the central stats object
    for s in project_with_sources.stats:
        if isinstance(s.data, ProfileProperties) and s.meta.organelle_id == "mito_0007":
            profile_stat = s
            break
    else:
        raise ValueError("Profile stat for mito_0007 not found in project stats")

    # .meta for context and .data for measurements
    assert profile_stat.meta.axis_used == "skeleton"
    assert len(profile_stat.data.perimeters) > 0
    assert len(profile_stat.data.widths) == len(profile_stat.data.perimeters)
    assert len(profile_stat.data.ratios) == len(profile_stat.data.perimeters)

    # Check first three slices
    expected_perimeters = [24.21223, 23.64194, 23.06106]
    expected_widths = [8.09431, 7.98912, 7.91002]
    expected_ratios = [0.3343066706371119, 0.337921507287473, 0.34300331381124716]

    np.testing.assert_almost_equal(
        profile_stat.data.perimeters[:3], expected_perimeters, decimal=5
    )
    np.testing.assert_almost_equal(
        profile_stat.data.widths[:3], expected_widths, decimal=5
    )
    np.testing.assert_almost_equal(
        profile_stat.data.ratios[:3], expected_ratios, decimal=5
    )
