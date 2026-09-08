"""Tests for hmarim (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmarim as mod


def test_hmarim_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
