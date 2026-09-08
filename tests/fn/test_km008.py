"""Tests for km008.kamath_ch2_attention_softmax_weights (re-fixtured from the doctest)."""

import doctest

import morie.fn.km008 as mod


def test_km008_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km008_edge():
    import pytest
    from morie.fn.km008 import kamath_ch2_attention_softmax_weights
    with pytest.raises(ValueError):
        kamath_ch2_attention_softmax_weights([])
