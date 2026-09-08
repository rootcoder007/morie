"""Tests for grbptt (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbptt as mod


def test_grbptt_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
