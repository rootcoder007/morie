"""Tests for bkngr.burkov_ngram_mle."""

from morie.fn.bkngr import burkov_ngram_mle


def test_bkngr_basic():
    assert burkov_ngram_mle(3, 4)["estimate"] == 0.75


def test_bkngr_edge():
    import pytest
    with pytest.raises(ValueError, match="undefined"):
        burkov_ngram_mle(0, 0)
