"""Tests for hmauxpt (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmauxpt as mod


def test_hmauxpt_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
