"""Tests for km017.kamath_ch2_ffn_relu (re-fixtured from the doctest)."""

import doctest

import morie.fn.km017 as mod


def test_km017_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km017_edge():
    import pytest
    from morie.fn.km017 import kamath_ch2_ffn_relu
    with pytest.raises(ValueError):
        kamath_ch2_ffn_relu([[1.0, 2.0]], [[1.0]], [[1.0]], [0.0], [0.0])
