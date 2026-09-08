"""Tests for grbpe (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbpe as mod


def test_grbpe_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
