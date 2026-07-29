"""Tests for grdeit (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grdeit as mod


def test_grdeit_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
