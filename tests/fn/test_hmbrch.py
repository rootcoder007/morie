"""Tests for hmbrch (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbrch as mod


def test_hmbrch_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
