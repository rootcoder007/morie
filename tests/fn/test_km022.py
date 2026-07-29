"""Tests for km022.kamath_ch2_mlm_loss (re-fixtured from the doctest)."""

import doctest

import morie.fn.km022 as mod


def test_km022_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km022_edge():
    import pytest
    from morie.fn.km022 import kamath_ch2_mlm_loss
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_mlm_loss(*([None] * 2))
