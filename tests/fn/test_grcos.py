"""Tests for grcos (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grcos as mod


def test_grcos_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
