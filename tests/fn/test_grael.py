"""Tests for grael (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grael as mod


def test_grael_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
