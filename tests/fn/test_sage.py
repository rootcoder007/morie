"""Tests for sage.graphsage."""

from morie.fn import _array_core as np

from morie.fn.sage import graphsage


def test_sage_basic():
    """Test basic functionality."""
    G = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = graphsage(G, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sage_edge():
    """Test edge cases."""
    G = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = graphsage(G, X)
    assert isinstance(result, dict)
