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
