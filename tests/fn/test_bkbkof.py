"""Tests for bkbkof.burkov_ngram_backoff."""

from morie.fn.bkbkof import burkov_ngram_backoff


def test_bkbkof_basic():
    out = burkov_ngram_backoff([(0, 5), (2, 8)], alpha=0.4)
    assert abs(out["estimate"] - 0.1) < 1e-12
    assert out["order_used"] == 1


def test_bkbkof_edge():
    import pytest
    with pytest.raises(ValueError, match="nowhere left"):
        burkov_ngram_backoff([(0, 5)])
