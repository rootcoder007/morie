"""Tests for hmbgdg (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbgdg as mod


def test_hmbgdg_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
