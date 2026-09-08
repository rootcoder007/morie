"""Tests for hmbert (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbert as mod


def test_hmbert_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
