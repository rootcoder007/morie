"""Tests for hmblp2 (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmblp2 as mod


def test_hmblp2_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
