"""Tests for hma3c (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hma3c as mod


def test_hma3c_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
