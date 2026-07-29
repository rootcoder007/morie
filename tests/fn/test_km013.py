"""Tests for km013.kamath_ch2_positional_encoding_sin (re-fixtured from the doctest)."""

import doctest

import morie.fn.km013 as mod


def test_km013_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km013_edge():
    import pytest
    from morie.fn.km013 import kamath_ch2_positional_encoding_sin
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_positional_encoding_sin(*([None] * 3))
