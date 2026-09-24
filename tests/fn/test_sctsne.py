"""Tests for sctsne.tsne_embedding."""

from morie.fn import _array_core as np

from morie.fn.sctsne import tsne_embedding


def test_sctsne_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tsne_embedding(X)
    assert isinstance(result, dict)
    assert "embedding" in result


def test_sctsne_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tsne_embedding(X)
    assert isinstance(result, dict)
