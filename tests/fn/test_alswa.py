"""Tests for alswa.alammar_sliding_window_attention."""

from morie.fn.alswa import alammar_sliding_window_attention


def test_alswa_basic():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    out = alammar_sliding_window_attention(X, X, X, 1)
    assert out["attention"][2][0] == 0.0
    assert out["attention"][2][2] == 1.0


def test_alswa_edge():
    import pytest
    with pytest.raises(ValueError, match="positive"):
        alammar_sliding_window_attention([[1.0]], [[1.0]], [[1.0]], 0)
