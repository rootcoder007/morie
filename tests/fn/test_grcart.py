"""Tests for grcart (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grcart as mod


def test_grcart_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
