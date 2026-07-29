"""Tests for km002.kamath_ch2_context_vector (re-fixtured from the doctest)."""

import doctest

import morie.fn.km002 as mod


def test_km002_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km002_edge():
    import pytest
    from morie.fn.km002 import kamath_ch2_context_vector
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_context_vector(*([None] * 2))
