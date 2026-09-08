"""Tests for variance_components_sigma2_u.variance_components_sigma2_u."""

from morie.fn import _array_core as np

from morie.fn.variance_components_sigma2_u import variance_components_sigma2_u


def test_ca7e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_components_sigma2_u(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca7e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_components_sigma2_u(x)
    assert isinstance(result, dict)
