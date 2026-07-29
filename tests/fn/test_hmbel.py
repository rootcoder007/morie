"""Tests for hmbel (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbel as mod


def test_hmbel_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
