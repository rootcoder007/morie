"""Tests for grddqn (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grddqn as mod


def test_grddqn_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
