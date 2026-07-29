"""Tests for km023.kamath_ch2_rtd_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km023 as mod


def test_km023_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km023_edge():
    import pytest
    from morie.fn.km023 import kamath_ch2_rtd_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_rtd_loss(*([None] * 2))
