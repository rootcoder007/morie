"""Tests for sgtrwl.sgt_random_walk_laplacian."""

from morie.fn import _array_core as np

from morie.fn.sgtrwl import sgt_random_walk_laplacian


def test_sgtrwl_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_random_walk_laplacian(W)
    assert isinstance(result, dict)
    assert "Lrw" in result


def test_sgtrwl_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_random_walk_laplacian(W)
    assert isinstance(result, dict)
