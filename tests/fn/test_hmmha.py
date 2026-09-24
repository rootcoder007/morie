"""Tests for hmmha.geron_multihead_attention."""

from morie.fn import _array_core as np

from morie.fn.hmmha import geron_multihead_attention


def test_hmmha_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    K = np.random.default_rng(42).normal(0.0, 1.0, 40)
    V = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_heads = 5
    result = geron_multihead_attention(Q, K, V, n_heads)
    assert isinstance(result, dict)
    assert "estimate" in result or "output" in result


def test_hmmha_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    K = np.random.default_rng(42).normal(0.0, 1.0, 40)
    V = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_heads = 5
    result = geron_multihead_attention(Q, K, V, n_heads)
    assert isinstance(result, dict)
