"""Tests for hrzsier.horowitz_series_regression."""

from morie.fn import _array_core as np

from morie.fn.hrzsier import horowitz_series_regression


def test_hrzsier_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_series_regression(x, y)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzsier_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_series_regression(x, y)
    assert isinstance(result, dict)
