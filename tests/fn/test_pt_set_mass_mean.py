"""Tests for pt_set_mass_mean.pt_set_mass_mean."""

import math

from morie.fn import _array_core as np

from morie.fn.pt_set_mass_mean import pt_set_mass_mean


def test_ghs028_basic():
    """Test basic functionality with a list of (alpha_taken, alpha_other) pairs."""
    rng = np.random.default_rng(42)
    alpha_epsilon = [(float(a), float(b)) for a, b in zip(
        rng.uniform(0.5, 5.0, 5), rng.uniform(0.5, 5.0, 5)
    )]
    result = pt_set_mass_mean(alpha_epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value" in result
    assert "second_moment" in result
    assert "variance" in result
    assert "method" in result
    assert 0.0 <= result["estimate"] <= 1.0
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["second_moment"])
    assert math.isfinite(result["variance"])
    assert result["variance"] >= 0.0
    assert result["second_moment"] >= result["estimate"] ** 2


def test_ghs028_edge():
    """Test edge case with a single pair and derive expected values from the docstring."""
    alpha_epsilon = [(2.0, 3.0)]
    result = pt_set_mass_mean(alpha_epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "second_moment" in result
    a_take, a_other = 2.0, 3.0
    s = a_take + a_other
    expected_mean = a_take / s
    expected_m2 = a_take * (a_take + 1.0) / (s * (s + 1.0))
    assert math.isclose(result["estimate"], expected_mean)
    assert math.isclose(result["second_moment"], expected_m2)
    assert math.isclose(result["variance"], expected_m2 - expected_mean ** 2)
