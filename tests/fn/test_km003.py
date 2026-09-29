"""Tests for km003.kamath_ch2_context_simplest (re-fixtured from the doctest)."""

import doctest

import morie.fn.km003 as mod


def test_km003_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km003_edge():
    import pytest

    from morie.fn.km003 import kamath_ch2_context_simplest

    with pytest.raises(ValueError):
        kamath_ch2_context_simplest([9.0], all_states=[[1.0]])


def test_simplest_context_agrees_with_the_last_state():
    r = mod.kamath_ch2_context_simplest([2.0, 8.0], all_states=[[1.0, 5.0], [2.0, 8.0]])
    assert r["context"] == [2.0, 8.0] and r["agrees_with_eq22"] is True
