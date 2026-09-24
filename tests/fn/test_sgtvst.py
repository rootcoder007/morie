"""Tests for sgtvst.sgt_vertex_strengths."""

from morie.fn import _array_core as np

from morie.fn.sgtvst import sgt_vertex_strengths


def test_sgtvst_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_vertex_strengths(W)
    assert isinstance(result, dict)
    assert "strength" in result


def test_sgtvst_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_vertex_strengths(W)
    assert isinstance(result, dict)
