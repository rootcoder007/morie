"""Tests for grbf16 (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbf16 as mod


def test_grbf16_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
