"""Tests for grbeam (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbeam as mod


def test_grbeam_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
