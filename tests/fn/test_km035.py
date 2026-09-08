"""Tests for km035.kamath_ch2_gpt_supervised_softmax (re-fixtured from the doctest)."""

import doctest

import morie.fn.km035 as mod


def test_km035_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km035_edge():
    import pytest
    from morie.fn.km035 import kamath_ch2_gpt_supervised_softmax
    with pytest.raises(ValueError):
        kamath_ch2_gpt_supervised_softmax("d", [1.0, 2.0], [[1.0]])
