"""Tests for grddim (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grddim as mod


def test_grddim_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
