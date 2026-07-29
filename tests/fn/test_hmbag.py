"""Tests for hmbag (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbag as mod


def test_hmbag_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
