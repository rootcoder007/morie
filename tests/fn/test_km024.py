"""Tests for km024.kamath_ch2_std_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km024 as mod


def test_km024_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km024_edge():
    import pytest
    from morie.fn.km024 import kamath_ch2_std_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_std_loss(*([None] * 2))
