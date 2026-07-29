"""Tests for km029.kamath_ch2_sbo_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km029 as mod


def test_km029_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km029_edge():
    import pytest
    from morie.fn.km029 import kamath_ch2_sbo_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_sbo_loss(*([None] * 2))
