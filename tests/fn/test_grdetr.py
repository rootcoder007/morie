"""Tests for grdetr (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grdetr as mod


def test_grdetr_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
