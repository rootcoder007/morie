"""Tests for alctf.alammar_c_tfidf."""

from morie.fn.alctf import alammar_c_tfidf


def test_alctf_basic():
    out = alammar_c_tfidf([[4.0, 0.0], [0.0, 4.0]])
    assert out["top_term_per_class"] == [0, 1]


def test_alctf_edge():
    import pytest
    with pytest.raises(ValueError, match="zero corpus frequency"):
        alammar_c_tfidf([[1.0, 0.0]], corpus_freq=[1.0, 0.0])
