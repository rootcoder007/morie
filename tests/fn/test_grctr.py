"""Tests for grctr (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grctr as mod


def test_grctr_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
