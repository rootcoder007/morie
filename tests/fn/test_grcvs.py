"""Tests for grcvs (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grcvs as mod


def test_grcvs_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
