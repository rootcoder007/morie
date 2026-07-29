"""Tests for hmbic (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbic as mod


def test_hmbic_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
