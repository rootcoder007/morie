"""Tests for grbic (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbic as mod


def test_grbic_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
