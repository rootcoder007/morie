"""Tests for hmbf16 (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbf16 as mod


def test_hmbf16_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
