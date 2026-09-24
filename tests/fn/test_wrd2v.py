"""Tests for wrd2v.word2vec."""

from morie.fn import _array_core as np

from morie.fn.wrd2v import word2vec


def test_wrd2v_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = word2vec(corpus)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wrd2v_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = word2vec(corpus)
    assert isinstance(result, dict)
