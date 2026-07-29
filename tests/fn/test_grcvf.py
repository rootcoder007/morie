"""Tests for grcvf (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grcvf as mod


def test_grcvf_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
