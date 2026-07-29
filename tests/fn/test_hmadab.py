"""Tests for hmadab (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmadab as mod


def test_hmadab_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
