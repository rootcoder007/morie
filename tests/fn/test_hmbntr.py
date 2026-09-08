"""Tests for hmbntr (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbntr as mod


def test_hmbntr_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
