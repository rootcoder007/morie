"""Tests for hmblip (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmblip as mod


def test_hmblip_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
