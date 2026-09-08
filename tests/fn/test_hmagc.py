"""Tests for hmagc (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmagc as mod


def test_hmagc_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
