"""Tests for km040.kamath_ch2_moe_topk_gating (re-fixtured from the doctest)."""

import doctest

import morie.fn.km040 as mod


def test_km040_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km040_edge():
    import pytest
    from morie.fn.km040 import kamath_ch2_moe_topk_gating
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_moe_topk_gating(*([None] * 3))
