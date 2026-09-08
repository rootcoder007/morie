"""Tests for hmaen (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmaen as mod


def test_hmaen_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
