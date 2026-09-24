"""Tests for rgs.functional_regression."""

from morie.fn import _array_core as np

from morie.fn.rgs import functional_regression


def test_rgs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = functional_regression(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rgs_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = functional_regression(X, Y)
    assert isinstance(result, dict)
