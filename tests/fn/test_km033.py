"""Tests for km033.kamath_ch2_dae_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km033 as mod


def test_km033_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km033_edge():
    import pytest
    from morie.fn.km033 import kamath_ch2_dae_loss
    with pytest.raises(ValueError):
        kamath_ch2_dae_loss([1.5], "x")
