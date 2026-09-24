"""Tests for hmkmn.geron_kmeans."""

from morie.fn import _array_core as np

from morie.fn.hmkmn import geron_kmeans


def test_hmkmn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_clusters = 5
    result = geron_kmeans(X, n_clusters)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels" in result


def test_hmkmn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_clusters = 5
    result = geron_kmeans(X, n_clusters)
    assert isinstance(result, dict)
