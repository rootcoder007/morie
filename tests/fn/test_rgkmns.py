"""Tests for rgkmns.rangayyan_kmeans."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_kmeans


def test_rgkmns_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = rangayyan_kmeans(X, k)
    assert isinstance(result, dict)
    assert "labels" in result


def test_rgkmns_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = rangayyan_kmeans(X, k)
    assert isinstance(result, dict)
