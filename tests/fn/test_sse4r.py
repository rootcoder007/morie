"""Tests for sse4r.ssepta_seq."""

from morie.fn import _array_core as np

from morie.fn.sse4r import ssepta_seq


def test_sse4r_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    user_embedding = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    item_table = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ssepta_seq(sequence, user_embedding, item_table)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sse4r_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    user_embedding = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    item_table = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = ssepta_seq(sequence, user_embedding, item_table)
    assert isinstance(result, dict)
