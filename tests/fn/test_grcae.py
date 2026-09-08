"""Tests for grcae (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grcae as mod


def test_grcae_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
