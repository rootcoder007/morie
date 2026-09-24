"""Tests for ragRet.rag_retrieval."""

from morie.fn import _array_core as np

from morie.fn.ragRet import rag_retrieval


def test_ragRet_basic():
    """Test basic functionality."""
    query = 0.5
    corpus = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rag_retrieval(query, corpus)
    assert isinstance(result, dict)
    assert "indices" in result


def test_ragRet_edge():
    """Test edge cases."""
    query = 0.5
    corpus = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rag_retrieval(query, corpus)
    assert isinstance(result, dict)
