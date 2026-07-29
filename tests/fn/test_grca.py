"""Tests for grca (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grca as mod


def test_grca_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0
