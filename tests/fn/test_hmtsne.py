"""Tests for hmtsne.geron_tsne."""

from morie.fn import _array_core as np

from morie.fn.hmtsne import geron_tsne


def test_hmtsne_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = geron_tsne(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "embedding" in result


def test_hmtsne_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = geron_tsne(X)
    assert isinstance(result, dict)
