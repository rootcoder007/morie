"""Tests for grpqa.grouped_query_attention."""

import numpy as np
import pytest

from morie.fn.grpqa import grouped_query_attention


def test_grpqa_shapes_and_row_stochastic_attention():
    n_heads, n_kv, seq_len, d_head = 8, 2, 5, 4
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(n_heads, seq_len, d_head))
    K = rng.normal(size=(n_kv, seq_len, d_head))
    V = rng.normal(size=(n_kv, seq_len, d_head))

    r = grouped_query_attention(Q, K, V, n_heads=n_heads, n_kv_heads=n_kv)
    out = np.asarray(r["tensor"], dtype=float)
    attn = np.asarray(r["attn"], dtype=float)
    assert out.shape == (n_heads, seq_len, d_head)
    assert attn.shape == (n_heads, seq_len, seq_len)
    np.testing.assert_allclose(attn.sum(axis=-1), 1.0, atol=1e-10)


def test_grpqa_heads_sharing_a_kv_group_see_the_same_keys():
    """Grouped-query attention exists to let several query heads share one KV
    head. Give two heads in the same group identical queries and their outputs
    must coincide; heads in different groups need not."""
    n_heads, n_kv, seq_len, d_head = 4, 2, 6, 3
    rng = np.random.default_rng(1)
    q = rng.normal(size=(seq_len, d_head))
    Q = np.stack([q, q, q, q])
    K = rng.normal(size=(n_kv, seq_len, d_head))
    V = rng.normal(size=(n_kv, seq_len, d_head))

    out = np.asarray(
        grouped_query_attention(Q, K, V, n_heads=n_heads, n_kv_heads=n_kv)["tensor"], dtype=float
    )
    # Heads 0,1 share KV group 0 and heads 2,3 share group 1.
    np.testing.assert_allclose(out[0], out[1], atol=1e-10)
    np.testing.assert_allclose(out[2], out[3], atol=1e-10)
    assert not np.allclose(out[0], out[2], atol=1e-6)


def test_grpqa_requires_heads_divisible_by_kv_heads():
    rng = np.random.default_rng(2)
    Q = rng.normal(size=(6, 4, 3))
    K = rng.normal(size=(4, 4, 3))
    V = rng.normal(size=(4, 4, 3))
    with pytest.raises(ValueError, match="multiple"):
        grouped_query_attention(Q, K, V, n_heads=6, n_kv_heads=4)
