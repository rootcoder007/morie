"""Tests for tfidfV.tfidf."""

from morie.fn import _array_core as np

from morie.fn.tfidfV import tfidf


def test_tfidfV_basic():
    """Test basic functionality."""
    docs = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tfidf(docs)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tfidfV_edge():
    """Test edge cases."""
    docs = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tfidf(docs)
    assert isinstance(result, dict)
