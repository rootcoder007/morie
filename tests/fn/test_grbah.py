"""Tests for grbah (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbah as mod


def test_grbah_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
