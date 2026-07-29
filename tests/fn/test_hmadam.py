"""Tests for hmadam (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmadam as mod


def test_hmadam_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
