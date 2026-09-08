"""Tests for hmbat (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbat as mod


def test_hmbat_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
