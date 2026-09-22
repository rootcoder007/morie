"""Tests for cfst.causal_forest."""

from morie.fn import _array_core as np

from morie.fn.cfst import causal_forest


def test_cfst_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_forest(Y, T, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cfst_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_forest(Y, T, X)
    assert isinstance(result, dict)
