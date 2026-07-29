"""Tests for hmbms (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbms as mod


def test_hmbms_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
