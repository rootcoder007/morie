"""Tests for altkemb.alammar_token_embedding_lookup."""

from morie.fn.altkemb import alammar_token_embedding_lookup


def test_altkemb_basic():
    out = alammar_token_embedding_lookup([1, 0], [[1.0, 2.0], [3.0, 4.0]])
    assert out["embeddings"] == [[3.0, 4.0], [1.0, 2.0]]


def test_altkemb_edge():
    import pytest
    with pytest.raises(ValueError, match="vocabulary"):
        alammar_token_embedding_lookup([9], [[1.0]])
