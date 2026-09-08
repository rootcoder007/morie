"""Tests for hmadgr (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmadgr as mod


def test_hmadgr_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
