"""Tests for aldocemb.alammar_document_embedding_pool."""

from morie.fn.aldocemb import alammar_document_embedding_pool


def test_aldocemb_basic():
    out = alammar_document_embedding_pool([[2.0], [4.0], [99.0]], [1, 1, 0])
    assert out["embedding"] == [3.0]


def test_aldocemb_edge():
    import pytest
    with pytest.raises(ValueError, match="all-padding"):
        alammar_document_embedding_pool([[1.0]], [0])
