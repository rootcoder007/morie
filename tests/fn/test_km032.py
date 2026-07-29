"""Tests for km032.kamath_ch2_seq2seq_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km032 as mod


def test_km032_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km032_edge():
    import pytest
    from morie.fn.km032 import kamath_ch2_seq2seq_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_seq2seq_loss(*([None] * 4))
