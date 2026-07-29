"""Tests for km004.kamath_ch2_decoder_hidden_state (re-fixtured from the doctest)."""

import doctest

import morie.fn.km004 as mod


def test_km004_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km004_edge():
    import pytest
    from morie.fn.km004 import kamath_ch2_decoder_hidden_state
    with pytest.raises(ValueError):
        kamath_ch2_decoder_hidden_state([1.0], [1.0], [1.0, 2.0])
