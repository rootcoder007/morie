"""Tests for hma2c (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hma2c as mod


def test_hma2c_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
