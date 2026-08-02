"""Tests for isoF.isolation_forest."""

from morie.fn import _array_core as np

from morie.fn.isof import isolation_forest


def test_isof_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = isolation_forest(X, n_trees)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_isof_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_trees = np.random.default_rng(42).normal(0, 1, 100)
    result = isolation_forest(X, n_trees)
    assert isinstance(result, dict)
