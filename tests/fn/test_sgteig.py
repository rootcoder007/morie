"""Tests for sgteig.sgt_eigenvector_centrality."""

from morie.fn import _array_core as np

from morie.fn.sgteig import sgt_eigenvector_centrality


def test_sgteig_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_eigenvector_centrality(A)
    assert isinstance(result, dict)
    assert "centrality" in result


def test_sgteig_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_eigenvector_centrality(A)
    assert isinstance(result, dict)
