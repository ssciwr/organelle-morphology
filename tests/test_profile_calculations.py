import numpy as np
from organelle_morphology.profile_calculations import ProfileCalculator, ProfileData


def test_calculate_profile_lengths(project_with_sources):
    """Regression-Test the 2D profile length calculation pipeline."""
    # Initialize the calculator with the project
    calculator = ProfileCalculator(project_with_sources)

    # Regression test using 'mito_0007' from the synthetic data fixture
    results = calculator.calculate_profile_lengths(
        ids="mito_0007", axis="z", num_slices=3
    )

    assert isinstance(results, dict)
    assert "mito_0007" in results

    # The dictionary now holds a ProfileData dataclass instance
    profile_data = results["mito_0007"]
    assert isinstance(profile_data, ProfileData)
    assert profile_data.organelle_id == "mito_0007"
    assert profile_data.axis_used == "z"
    assert profile_data.num_slices_attempted == 3

    # Extract the perimeters list from the dataclass
    perimeters = profile_data.perimeters

    # Expected values from the synthetic data fixture
    expected_perimeters = [33.53122292, 39.50609665, 29.67766953]
    assert len(perimeters) == len(expected_perimeters)
    np.testing.assert_almost_equal(perimeters, expected_perimeters, decimal=5)

    # Verify the mean property is calculating correctly
    expected_mean = np.mean(expected_perimeters)
    np.testing.assert_almost_equal(
        profile_data.mean_perimeter, expected_mean, decimal=5
    )


def test_calculate_random_profiles(project_with_sources):
    """Regression-Test the 2D profile length with random planes."""

    calculator = ProfileCalculator(project_with_sources)
    results = calculator.calculate_random_profiles(
        ids="mito_0007", num_planes=5, seed=42
    )

    perimeters = results["mito_0007"].perimeters
    expected = [31.36676084, 52.20270082, 32.24204422, 22.81741491, 30.23820865]

    np.testing.assert_almost_equal(perimeters, expected, decimal=5)


def test_calculate_skeleton_profiles(project_with_sources):
    """Regression-Test the 2D profile length with skeleton profiles."""
    # Generate the skeleton first
    project_with_sources.skeletonize_wavefront(ids="mito_0007")

    calculator = ProfileCalculator(project_with_sources)
    results = calculator.calculate_skeleton_profiles(ids="mito_0007")

    profile_data = results["mito_0007"]

    assert profile_data.axis_used == "skeleton"
    assert len(profile_data.perimeters) > 0
    assert len(profile_data.perimeters) == len(profile_data.widths)

    # Check only the first three slices
    expected_perimeters = [24.21223, 23.64194, 23.06106]
    expected_widths = [8.09431, 7.98912, 7.91002]

    np.testing.assert_almost_equal(
        profile_data.perimeters[:3], expected_perimeters, decimal=5
    )
    np.testing.assert_almost_equal(profile_data.widths[:3], expected_widths, decimal=5)
