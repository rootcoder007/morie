"""Tests for hmbp (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbp as mod


def test_hmbp_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
