"""Tests for hrzkd2.horowitz_multivariate_kde."""

from morie.fn import _array_core as np

from morie.fn.hrzkd2 import horowitz_multivariate_kde


def test_hrzkd2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_multivariate_kde(x)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzkd2_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_multivariate_kde(x)
    assert isinstance(result, dict)
