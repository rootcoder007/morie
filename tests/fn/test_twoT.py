"""Tests for twoT.two_tower."""

from morie.fn import _array_core as np

from morie.fn.twoT import two_tower


def test_twoT_basic():
    """Test basic functionality."""
    query_embedding = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    item_embeddings = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = two_tower(query_embedding, item_embeddings)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_twoT_edge():
    """Test edge cases."""
    query_embedding = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    item_embeddings = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = two_tower(query_embedding, item_embeddings)
    assert isinstance(result, dict)
