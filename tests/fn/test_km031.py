"""Tests for km031.kamath_ch2_sop_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km031 as mod


def test_km031_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km031_edge():
    import pytest
    from morie.fn.km031 import kamath_ch2_sop_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_sop_loss(*([None] * 3))
