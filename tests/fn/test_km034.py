"""Tests for km034.kamath_ch2_gpt_unsupervised_obj (re-fixtured from the doctest)."""

import doctest

import morie.fn.km034 as mod


def test_km034_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km034_edge():
    import pytest
    from morie.fn.km034 import kamath_ch2_gpt_unsupervised_obj
    with pytest.raises(ValueError):
        kamath_ch2_gpt_unsupervised_obj([2.0])
