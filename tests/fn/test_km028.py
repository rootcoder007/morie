"""Tests for km028.kamath_ch2_alm_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km028 as mod


def test_km028_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km028_edge():
    import pytest
    from morie.fn.km028 import kamath_ch2_alm_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_alm_loss(*([None] * 2))
