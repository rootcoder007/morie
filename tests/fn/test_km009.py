"""Tests for km009.kamath_ch2_softmax_element (re-fixtured from the doctest)."""

import doctest

import morie.fn.km009 as mod


def test_km009_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km009_edge():
    import pytest
    from morie.fn.km009 import kamath_ch2_softmax_element
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_softmax_element(*([None] * 2))
