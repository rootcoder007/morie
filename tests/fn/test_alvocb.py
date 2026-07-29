"""Tests for alvocb.alammar_tokenizer_vocab_overlap."""

from morie.fn.alvocb import alammar_tokenizer_vocab_overlap


def test_alvocb_basic():
    out = alammar_tokenizer_vocab_overlap(["a", "b"], ["b", "c"])
    assert abs(out["estimate"] - 1 / 3) < 1e-12


def test_alvocb_edge():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        alammar_tokenizer_vocab_overlap([], [])
