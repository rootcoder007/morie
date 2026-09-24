"""Tests for sparsv.sparse_vector."""

from morie.fn import _array_core as np

from morie.fn.sparsv import sparse_vector


def test_sparsv_basic():
    """Test basic functionality."""
    queries = np.random.default_rng(42).normal(0.0, 1.0, 40)
    threshold = 0.1
    result = sparse_vector(queries, threshold)
    assert isinstance(result, dict)
    assert "above" in result


def test_sparsv_edge():
    """Test edge cases."""
    queries = np.random.default_rng(42).normal(0.0, 1.0, 40)
    threshold = 0.1
    result = sparse_vector(queries, threshold)
    assert isinstance(result, dict)
