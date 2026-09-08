"""Tests for grarma (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grarma as mod


def test_grarma_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
