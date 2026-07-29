"""Tests for hmadmw (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmadmw as mod


def test_hmadmw_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
