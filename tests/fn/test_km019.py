"""Tests for km019.kamath_ch2_masked_attention (re-fixtured from the doctest)."""

import doctest

import morie.fn.km019 as mod


def test_km019_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km019_edge():
    import pytest
    from morie.fn.km019 import kamath_ch2_masked_attention
    with pytest.raises(ValueError):
        kamath_ch2_masked_attention([[1.0]], [[1.0, 2.0]], [[1.0]], [[0.0]])
