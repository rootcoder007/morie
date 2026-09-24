"""Tests for kmdpr.kamath_dense_passage_retrieval."""

from morie.fn import _array_core as np

from morie.fn.kmdpr import kamath_dense_passage_retrieval


def test_kmdpr_basic():
    """Test basic functionality."""
    q_embed = [1.0, 0.0]
    p_embeds = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]
    k = 2
    result = kamath_dense_passage_retrieval(q_embed, p_embeds, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmdpr_edge():
    """Test edge cases."""
    q_embed = [1.0, 0.0]
    p_embeds = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]
    k = 2
    result = kamath_dense_passage_retrieval(q_embed, p_embeds, k)
    assert isinstance(result, dict)
