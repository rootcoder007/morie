"""Tests for hmbrnn (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbrnn as mod


def test_hmbrnn_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
