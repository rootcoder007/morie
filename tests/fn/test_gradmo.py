"""Tests for gradmo (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.gradmo as mod


def test_gradmo_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
