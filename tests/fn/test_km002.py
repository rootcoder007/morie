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


def test_context_mappings_recomputed():
    H = [[1.0, 5.0], [3.0, 2.0], [2.0, 8.0]]
    assert mod.kamath_ch2_context_vector(H)["context"] == [2.0, 5.0]
    assert mod.kamath_ch2_context_vector(H, "last")["context"] == [2.0, 8.0]
    assert mod.kamath_ch2_context_vector(H, "max")["context"] == [3.0, 8.0]
    assert mod.kamath_ch2_context_vector(H, lambda A: A[0] + A[1])["context"] == [4.0, 7.0]
