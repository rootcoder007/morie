"""Tests for hmbart (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbart as mod


def test_hmbart_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
