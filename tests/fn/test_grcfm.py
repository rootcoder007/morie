"""Tests for grcfm (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grcfm as mod


def test_grcfm_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
