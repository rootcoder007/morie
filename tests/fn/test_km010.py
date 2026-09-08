"""Tests for km010.kamath_ch2_attention_output (re-fixtured from the doctest)."""

import doctest

import morie.fn.km010 as mod


def test_km010_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km010_edge():
    import pytest
    from morie.fn.km010 import kamath_ch2_attention_output
    with pytest.raises(ValueError):
        kamath_ch2_attention_output([0.5], [[1.0], [2.0]])
