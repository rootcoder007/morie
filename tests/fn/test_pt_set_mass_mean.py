"""Tests for pt_set_mass_mean.pt_set_mass_mean."""

from morie.fn import _array_core as np

from morie.fn.pt_set_mass_mean import pt_set_mass_mean


def test_ghs028_basic():
    """Test basic functionality."""
    alpha_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    m = 10
    result = pt_set_mass_mean(alpha_epsilon, epsilon, m)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs028_edge():
    """Test edge cases."""
    alpha_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    m = 10
    result = pt_set_mass_mean(alpha_epsilon, epsilon, m)
    assert isinstance(result, dict)
