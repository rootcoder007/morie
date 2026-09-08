"""Tests for altkp.alammar_tokenization_pipeline."""

from morie.fn.altkp import alammar_tokenization_pipeline


def test_altkp_basic():
    v = ["[CLS]", "[SEP]", "[UNK]", "un", "##happy", "dog"]
    out = alammar_tokenization_pipeline("Unhappy dog", v)
    assert out["tokens"] == ["[CLS]", "un", "##happy", "dog", "[SEP]"]


def test_altkp_edge():
    import pytest
    with pytest.raises(ValueError, match="UNK"):
        alammar_tokenization_pipeline("x", ["[CLS]", "[SEP]", "x"])
