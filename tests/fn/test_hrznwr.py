"""Tests for hrznwr.horowitz_nw_regression."""

from morie.fn import _array_core as np

from morie.fn.hrznwr import horowitz_nw_regression


def test_hrznwr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nw_regression(x, y)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrznwr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nw_regression(x, y)
    assert isinstance(result, dict)
