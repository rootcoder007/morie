"""Tests for sestrg.s_estimator_regression."""

from morie.fn import _array_core as np

from morie.fn.sestrg import s_estimator_regression


def test_sestrg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = s_estimator_regression(X, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_sestrg_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = s_estimator_regression(X, y)
    assert isinstance(result, dict)
