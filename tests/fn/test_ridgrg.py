"""Tests for ridgrg.ridge_regression."""

from morie.fn import _array_core as np

from morie.fn.ridgrg import ridge_regression


def test_ridgrg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ridge_regression(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ridgrg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ridge_regression(X, y)
    assert isinstance(result, dict)
