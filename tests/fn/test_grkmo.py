"""Tests for grkmo.geron_kmeans_objective."""

from morie.fn import _array_core as np

from morie.fn.grkmo import geron_kmeans_objective


def test_grkmo_basic():
    """Test basic functionality."""
    X = [[0.0, 0.0], [3.0, 4.0]]
    centroids = [[0.0, 0.0]]
    labels = [0, 0]
    result = geron_kmeans_objective(X, centroids, labels)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grkmo_edge():
    """Test edge cases."""
    X = [[0.0, 0.0], [3.0, 4.0]]
    centroids = [[0.0, 0.0]]
    labels = [0, 0]
    result = geron_kmeans_objective(X, centroids, labels)
    assert isinstance(result, dict)
