"""Tests for hmbdn (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbdn as mod


def test_hmbdn_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
