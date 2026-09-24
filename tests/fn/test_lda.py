"""Tests for lda.lda_topic."""

from morie.fn import _array_core as np
from morie.fn.lda import lda_topic


def test_lda_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    vocab_size = 50
    docs_mat = rng.integers(0, vocab_size, (10, 20))
    docs = [[int(v) for v in row] for row in docs_mat]
    K = 5
    alpha = 0.1
    beta = 0.1
    result = lda_topic(docs, K, vocab_size, alpha, beta)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_lda_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    vocab_size = 20
    docs_mat = rng.integers(0, vocab_size, (5, 8))
    docs = [[int(v) for v in row] for row in docs_mat]
    K = 2
    alpha = 0.1
    beta = 0.1
    result = lda_topic(docs, K, vocab_size, alpha, beta)
    assert isinstance(result, dict)
    assert len(result) > 0
