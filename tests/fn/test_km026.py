"""Tests for km026.kamath_ch2_slm_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km026 as mod


def test_km026_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km026_edge():
    import pytest
    from morie.fn.km026 import kamath_ch2_slm_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_slm_loss(*([None] * 2))
