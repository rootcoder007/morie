"""Tests for almqa.alammar_multi_query_attention."""

from morie.fn.almqa import alammar_multi_query_attention


def test_almqa_basic():
    Q = [[[1.0, 0.0]], [[0.0, 1.0]]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0], [0.0]]
    out = alammar_multi_query_attention(Q, K, V, 2)
    assert out["kv_cache_ratio"] == 0.5


def test_almqa_edge():
    import pytest
    with pytest.raises(ValueError, match="query heads"):
        alammar_multi_query_attention([[[1.0]]], [[1.0]], [[1.0]], 2)
