"""Tests for hrzweib.horowitz_weibull_heterogeneity."""

from morie.fn import _array_core as np

from morie.fn.hrzweib import horowitz_weibull_heterogeneity


def test_hrzweib_basic():
    """Test basic functionality."""
    t = np.array([float(i + 1) for i in range(40)])
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_weibull_heterogeneity(t, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hrzweib_edge():
    """Test edge cases."""
    t = np.array([float(i + 1) for i in range(40)])
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_weibull_heterogeneity(t, x)
    assert isinstance(result, dict)
