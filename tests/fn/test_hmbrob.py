"""Tests for hmbrob (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbrob as mod


def test_hmbrob_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
