"""Tests for km011.kamath_ch2_scaled_dot_score (re-fixtured from the doctest)."""

import doctest

import morie.fn.km011 as mod


def test_km011_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km011_edge():
    import pytest
    from morie.fn.km011 import kamath_ch2_scaled_dot_score
    with pytest.raises(ValueError):
        kamath_ch2_scaled_dot_score([1.0], [1.0], d_k=9)
