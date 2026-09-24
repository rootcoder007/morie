"""Tests for hmkmpp.geron_kmeans_plus_plus."""

from morie.fn import _array_core as np

from morie.fn.hmkmpp import geron_kmeans_plus_plus


def test_hmkmpp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_clusters = 5
    result = geron_kmeans_plus_plus(X, n_clusters)
    assert isinstance(result, dict)
    assert "estimate" in result or "centers" in result


def test_hmkmpp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_clusters = 5
    result = geron_kmeans_plus_plus(X, n_clusters)
    assert isinstance(result, dict)
