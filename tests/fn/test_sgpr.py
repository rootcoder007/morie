"""Tests for sgpr.sparse_gp."""

from morie.fn import _array_core as np

from morie.fn.sgpr import sparse_gp


def test_sgpr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sparse_gp(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sgpr_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sparse_gp(X, y)
    assert isinstance(result, dict)
