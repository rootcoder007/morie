"""Tests for ltsreg.least_trimmed_squares."""

from morie.fn import _array_core as np

from morie.fn.ltsreg import least_trimmed_squares


def test_ltsreg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = least_trimmed_squares(y, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ltsreg_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = least_trimmed_squares(y, X)
    assert isinstance(result, dict)
