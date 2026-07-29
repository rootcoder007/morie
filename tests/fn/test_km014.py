"""Tests for km014.kamath_ch2_positional_encoding_cos (re-fixtured from the doctest)."""

import doctest

import morie.fn.km014 as mod


def test_km014_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km014_edge():
    import pytest
    from morie.fn.km014 import kamath_ch2_positional_encoding_cos
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_positional_encoding_cos(*([None] * 3))
