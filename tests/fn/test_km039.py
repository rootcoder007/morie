"""Tests for km039.kamath_ch2_moe_output (re-fixtured from the doctest)."""

import doctest

import morie.fn.km039 as mod


def test_km039_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km039_edge():
    import pytest
    from morie.fn.km039 import kamath_ch2_moe_output
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_moe_output(*([None] * 3))
