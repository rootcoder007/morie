"""Tests for hmagrd (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmagrd as mod


def test_hmagrd_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
