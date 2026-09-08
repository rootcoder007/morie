"""Tests for km020.kamath_ch2_ssl_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km020 as mod


def test_km020_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km020_edge():
    import pytest
    from morie.fn.km020 import kamath_ch2_ssl_loss
    with pytest.raises(ValueError):
        kamath_ch2_ssl_loss([])
