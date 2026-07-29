"""Tests for grbgd (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbgd as mod


def test_grbgd_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
