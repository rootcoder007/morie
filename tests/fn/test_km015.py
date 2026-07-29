"""Tests for km015.kamath_ch2_multihead_head_i (re-fixtured from the doctest)."""

import doctest

import morie.fn.km015 as mod


def test_km015_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km015_edge():
    import pytest
    from morie.fn.km015 import kamath_ch2_multihead_head_i
    with pytest.raises(ValueError):
        kamath_ch2_multihead_head_i([[1.0]], [[1.0]], [[1.0]], [[1.0, 2.0], [3.0, 4.0]], [[1.0]], [[1.0]])
