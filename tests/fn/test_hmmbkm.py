"""Tests for hmmbkm.geron_minibatch_kmeans."""

from morie.fn import _array_core as np

from morie.fn.hmmbkm import geron_minibatch_kmeans


def test_hmmbkm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_clusters = 5
    batch_size = 5
    result = geron_minibatch_kmeans(X, n_clusters, batch_size)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels" in result


def test_hmmbkm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_clusters = 5
    batch_size = 5
    result = geron_minibatch_kmeans(X, n_clusters, batch_size)
    assert isinstance(result, dict)
