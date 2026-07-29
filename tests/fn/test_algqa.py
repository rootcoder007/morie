"""Tests for algqa.alammar_grouped_query_attention."""

from morie.fn.algqa import alammar_grouped_query_attention


def test_algqa_basic():
    Q = [[[1.0, 0.0]], [[0.0, 1.0]]]
    K = [[[1.0, 0.0], [0.0, 1.0]]]
    V = [[[1.0], [0.0]]]
    out = alammar_grouped_query_attention(Q, K, V, 2, 1)
    assert out["kv_cache_ratio"] == 0.5


def test_algqa_edge():
    import pytest
    with pytest.raises(ValueError, match="divisible"):
        alammar_grouped_query_attention([[[1.0]]] * 3, [[[1.0]]] * 2,
                                        [[[1.0]]] * 2, 3, 2)
