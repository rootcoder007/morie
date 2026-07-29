"""Tests for grdae (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grdae as mod


def test_grdae_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
