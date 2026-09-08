"""Tests for graic (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.graic as mod


def test_graic_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
