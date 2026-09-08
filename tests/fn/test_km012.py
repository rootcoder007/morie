"""Tests for km012.kamath_ch2_scaled_dot_attention (re-fixtured from the doctest)."""

import doctest

import morie.fn.km012 as mod


def test_km012_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km012_edge():
    import pytest
    from morie.fn.km012 import kamath_ch2_scaled_dot_attention
    with pytest.raises(ValueError):
        kamath_ch2_scaled_dot_attention([[1.0]], [[1.0]], [[1.0]], d_k=9)
