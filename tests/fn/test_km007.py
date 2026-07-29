"""Tests for km007.kamath_ch2_attention_score (re-fixtured from the doctest)."""

import doctest

import morie.fn.km007 as mod


def test_km007_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km007_edge():
    import pytest
    from morie.fn.km007 import kamath_ch2_attention_score
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_attention_score(*([None] * 3))
