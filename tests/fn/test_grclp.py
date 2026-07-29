"""Tests for grclp (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grclp as mod


def test_grclp_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
