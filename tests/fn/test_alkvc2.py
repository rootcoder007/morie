"""Tests for alkvc2.alammar_kv_cache_lookup."""

from morie.fn.alkvc2 import alammar_kv_cache_lookup


def test_alkvc2_basic():
    out = alammar_kv_cache_lookup(None, None, [[1.0, 0.0]],
                                  [[2.0]], [[1.0, 0.0]])
    assert out["cache_length"] == 1
    assert out["output"] == [2.0]


def test_alkvc2_edge():
    out = alammar_kv_cache_lookup([[1.0, 0.0]], [[1.0]], [[0.0, 1.0]],
                                  [[3.0]], [[0.0, 1.0]])
    assert out["cache_length"] == 2
