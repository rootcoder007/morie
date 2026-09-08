"""Tests for alctxemb.alammar_contextualized_embedding."""

from morie.fn.alctxemb import alammar_contextualized_embedding


def test_alctxemb_basic():
    stack = [[[1.0], [2.0]], [[3.0], [4.0]]]
    out = alammar_contextualized_embedding(stack, -1, 1)
    assert out["embedding"] == [4.0]


def test_alctxemb_edge():
    import pytest
    with pytest.raises(ValueError, match="out of range"):
        alammar_contextualized_embedding([[[1.0]]], 5, 0)
