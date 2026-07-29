"""Tests for km041.kamath_ch2_mixtral_swiglu_moe (re-fixtured from the doctest)."""

import doctest

import morie.fn.km041 as mod


def test_km041_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km041_edge():
    import pytest
    from morie.fn.km041 import kamath_ch2_mixtral_swiglu_moe
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_mixtral_swiglu_moe(*([None] * 3))
