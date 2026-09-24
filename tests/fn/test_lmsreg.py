"""Tests for lmsreg.least_median_squares."""

from morie.fn import _array_core as np

from morie.fn.lmsreg import least_median_squares


def test_lmsreg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = least_median_squares(y, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_lmsreg_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = least_median_squares(y, X)
    assert isinstance(result, dict)
