"""Tests for km016.kamath_ch2_multihead_concat (re-fixtured from the doctest)."""

import doctest

import morie.fn.km016 as mod


def test_km016_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km016_edge():
    import pytest
    from morie.fn.km016 import kamath_ch2_multihead_concat
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_multihead_concat(*([None] * 2))
