"""Tests for km005.kamath_ch2_decoder_token_distribution (re-fixtured from the doctest)."""

import doctest

import morie.fn.km005 as mod


def test_km005_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km005_edge():
    import pytest
    from morie.fn.km005 import kamath_ch2_decoder_token_distribution
    with pytest.raises(ValueError):
        kamath_ch2_decoder_token_distribution([1.0], [1.0], [1.0], W=[[1.0]])
