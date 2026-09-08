"""Tests for grauc (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grauc as mod


def test_grauc_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
