"""Tests for km025.kamath_ch2_rts_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km025 as mod


def test_km025_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km025_edge():
    import pytest
    from morie.fn.km025 import kamath_ch2_rts_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_rts_loss(*([None] * 2))
