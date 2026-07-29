"""Tests for graut (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.graut as mod


def test_graut_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
