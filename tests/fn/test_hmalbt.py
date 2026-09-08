"""Tests for hmalbt (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmalbt as mod


def test_hmalbt_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
