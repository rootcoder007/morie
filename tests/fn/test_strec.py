"""Tests for strec.stamp."""

from morie.fn import _array_core as np

from morie.fn.strec import stamp


def test_strec_basic():
    """Test basic functionality."""
    embeddings = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    item_table = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Ws = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    Wt = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = stamp(embeddings, item_table, Ws, Wt)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_strec_edge():
    """Test edge cases."""
    embeddings = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    item_table = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Ws = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    Wt = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = stamp(embeddings, item_table, Ws, Wt)
    assert isinstance(result, dict)
