import numpy as np
from organelle_morphology.profile_calculations import ProfileCalculator, ProfileData


def test_calculate_profile_lengths(project_with_sources):
    """Test the 2D profile length calculation pipeline."""
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
