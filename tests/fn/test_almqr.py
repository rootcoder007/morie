"""Tests for almqr.alammar_multi_query_retrieval."""

from morie.fn.almqr import alammar_multi_query_retrieval


def test_almqr_basic():
    corpus = {"q": [1], "q0": [2]}
    out = alammar_multi_query_retrieval("q", 1, lambda q: corpus[q],
                                        lambda q, i: f"{q}{i}")
    assert out["documents"] == [1, 2]


def test_almqr_edge():
    import pytest
    with pytest.raises(ValueError, match="callable"):
        alammar_multi_query_retrieval("q", 1, None, None)
