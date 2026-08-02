"""Tests for nchunk.causal_chunked_attention."""

from morie.fn import _array_core as np
import pytest

from morie.fn.nchunk import causal_chunked_attention


def test_nchunk_basic():
    rng = np.random.default_rng(42)
    L, d = 12, 4
    Q, K, V = rng.normal(size=(L, d)), rng.normal(size=(L, d)), rng.normal(size=(L, 3))
    full = causal_chunked_attention(Q, K, V, chunk_size=4)
    s = Q @ K.T / np.sqrt(d)
    s = np.where(np.tril(np.ones((L, L), bool)), s, -np.inf)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    assert full["output"] == pytest.approx((e / e.sum(axis=1, keepdims=True)) @ V)


def test_nchunk_edge():
    rng = np.random.default_rng(0)
    Q = K = rng.normal(size=(8, 2))
    V = rng.normal(size=(8, 2))
    local = causal_chunked_attention(Q, K, V, chunk_size=4, n_chunks_back=0)
    assert not local["mask"][5, 1]  # chunk 1 cannot see chunk 0
    with pytest.raises(ValueError):
        causal_chunked_attention(Q, K, V, chunk_size=0)
