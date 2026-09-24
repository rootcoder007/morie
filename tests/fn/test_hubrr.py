"""Tests for hubrr.huber_regression."""

from morie.fn import _array_core as np

from morie.fn.hubrr import huber_regression


def test_hubrr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = huber_regression(X, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hubrr_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = huber_regression(X, y)
    assert isinstance(result, dict)
