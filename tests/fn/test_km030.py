"""Tests for km030.kamath_ch2_nsp_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km030 as mod


def test_km030_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km030_edge():
    import pytest
    from morie.fn.km030 import kamath_ch2_nsp_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_nsp_loss(*([None] * 3))
