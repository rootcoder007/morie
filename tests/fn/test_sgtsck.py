"""Tests for sgtsck.sgt_spectral_clustering_k."""

from morie.fn import _array_core as np

from morie.fn.sgtsck import sgt_spectral_clustering_k


def test_sgtsck_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_spectral_clustering_k(A)
    assert isinstance(result, dict)
    assert "labels" in result


def test_sgtsck_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_spectral_clustering_k(A)
    assert isinstance(result, dict)
