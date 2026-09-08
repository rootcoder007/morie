"""Tests for km018.kamath_ch2_layer_norm (re-fixtured from the doctest)."""

import doctest

import morie.fn.km018 as mod


def test_km018_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km018_edge():
    import pytest
    from morie.fn.km018 import kamath_ch2_layer_norm
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_layer_norm(*([None] * 5))
