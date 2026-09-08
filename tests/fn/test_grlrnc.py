"""Tests for grlrnc.geron_learning_curves."""

from morie.fn import _array_core as np

from morie.fn.grlrnc import geron_learning_curves


def test_grlrnc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_splits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_learning_curves(X, y, n_splits)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlrnc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_splits = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_learning_curves(X, y, n_splits)
    assert isinstance(result, dict)
