"""Tests for gradaw (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.gradaw as mod


def test_gradaw_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
