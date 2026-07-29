"""Tests for hmbv (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbv as mod


def test_hmbv_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
