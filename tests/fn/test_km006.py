"""Tests for km006.kamath_ch2_seq2seq_cross_entropy (re-fixtured from the doctest)."""

import doctest

import morie.fn.km006 as mod


def test_km006_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km006_edge():
    import pytest
    from morie.fn.km006 import kamath_ch2_seq2seq_cross_entropy
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_seq2seq_cross_entropy(*([None] * 3))
