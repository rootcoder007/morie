"""Tests for km036.kamath_ch2_gpt_supervised_obj (re-fixtured from the doctest)."""

import doctest

import morie.fn.km036 as mod


def test_km036_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km036_edge():
    import pytest
    from morie.fn.km036 import kamath_ch2_gpt_supervised_obj
    with pytest.raises(ValueError):
        kamath_ch2_gpt_supervised_obj([1.5])
