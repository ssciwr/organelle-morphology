import numpy as np
from organelle_morphology.profile_calculations import calculate_profile_lengths


def test_calculate_profile_lengths(project_with_sources):
    """Test the 2D profile length calculation pipeline."""
    # Regression test using 'mito_0007' from the synthetic data fixture
    results = calculate_profile_lengths(
        project_with_sources, ids="mito_0007", axis="z", num_slices=3
    )

    assert isinstance(results, dict)
    assert "mito_0007" in results
    perimeters = results["mito_0007"]

    # Expected values from the synthetic data fixture
    expected_perimeters = [33.53122292, 39.50609665, 29.67766953]
    assert len(perimeters) == len(expected_perimeters)
    np.testing.assert_almost_equal(perimeters, expected_perimeters, decimal=5)
