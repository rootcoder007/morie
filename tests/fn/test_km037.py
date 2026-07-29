"""Tests for km037.kamath_ch2_gpt_combined_obj (re-fixtured from the doctest)."""

import doctest

import morie.fn.km037 as mod


def test_km037_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km037_edge():
    import pytest
    from morie.fn.km037 import kamath_ch2_gpt_combined_obj
    with pytest.raises((ValueError, TypeError)):
        kamath_ch2_gpt_combined_obj(*([None] * 3))
