"""Tests for grac (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grac as mod


def test_grac_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
