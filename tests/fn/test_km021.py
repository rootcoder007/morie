"""Tests for km021.kamath_ch2_clm_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km021 as mod


def test_km021_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km021_edge():
    import pytest
    from morie.fn.km021 import kamath_ch2_clm_loss
    with pytest.raises(ValueError):
        kamath_ch2_clm_loss([1.5])
