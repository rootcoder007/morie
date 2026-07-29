"""Tests for hmalex (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmalex as mod


def test_hmalex_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
