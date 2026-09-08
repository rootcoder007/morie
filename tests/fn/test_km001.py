"""Tests for km001.kamath_ch2_unidirectional_encoder_state (re-fixtured from the doctest)."""

import doctest

import morie.fn.km001 as mod


def test_km001_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km001_edge():
    import pytest
    from morie.fn.km001 import kamath_ch2_unidirectional_encoder_state
    with pytest.raises(ValueError):
        kamath_ch2_unidirectional_encoder_state([1.0], [1.0, 2.0])
