"""Tests for hmbpet (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbpet as mod


def test_hmbpet_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
