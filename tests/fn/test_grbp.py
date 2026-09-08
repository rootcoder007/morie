"""Tests for grbp (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbp as mod


def test_grbp_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
