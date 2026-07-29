"""Tests for km027.kamath_ch2_tlm_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km027 as mod


def test_km027_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km027_edge():
    import pytest
    from morie.fn.km027 import kamath_ch2_tlm_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_tlm_loss(*([None] * 4))
