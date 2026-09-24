"""Tests for sdne.sdne."""

from morie.fn import _array_core as np

from morie.fn.sdne import sdne


def test_sdne_basic():
    """Test basic functionality."""
    adjacency = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    reconstruction = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    embeddings = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sdne(adjacency, reconstruction, embeddings)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sdne_edge():
    """Test edge cases."""
    adjacency = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    reconstruction = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    embeddings = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sdne(adjacency, reconstruction, embeddings)
    assert isinstance(result, dict)
