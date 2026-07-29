"""Tests for gr1cy (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.gr1cy as mod


def test_gr1cy_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
