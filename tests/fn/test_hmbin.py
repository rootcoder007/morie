"""Tests for hmbin (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbin as mod


def test_hmbin_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
