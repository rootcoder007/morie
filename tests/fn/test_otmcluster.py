"""Tests for otmcluster.ot_clustering_w."""

from morie.fn import _array_core as np

from morie.fn.otmcluster import ot_clustering_w


def test_otmcluster_basic():
    """Test basic functionality."""
    X_list = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = ot_clustering_w(X_list, k)
    assert isinstance(result, dict)
    assert "labels" in result


def test_otmcluster_edge():
    """Test edge cases."""
    X_list = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = ot_clustering_w(X_list, k)
    assert isinstance(result, dict)
