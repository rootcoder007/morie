"""Tests for grbrnn (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbrnn as mod


def test_grbrnn_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
