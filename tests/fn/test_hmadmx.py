"""Tests for hmadmx (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmadmx as mod


def test_hmadmx_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
