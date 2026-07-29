"""Tests for hmbnm (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbnm as mod


def test_hmbnm_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
