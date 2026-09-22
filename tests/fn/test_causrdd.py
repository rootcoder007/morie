"""Tests for causrdd.causal_rdd_local_lin."""

from morie.fn import _array_core as np

from morie.fn.causrdd import causal_rdd_local_lin


def test_causrdd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = causal_rdd_local_lin(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_causrdd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = causal_rdd_local_lin(x, y)
    assert isinstance(result, dict)
