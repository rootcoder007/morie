"""Tests for hmspcl.geron_spectral_clustering."""

from morie.fn import _array_core as np

from morie.fn.hmspcl import geron_spectral_clustering


def test_hmspcl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_spectral_clustering(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels" in result


def test_hmspcl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_spectral_clustering(X)
    assert isinstance(result, dict)
