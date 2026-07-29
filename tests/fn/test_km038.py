"""Tests for km038.kamath_ch2_gpt2_task_conditioning (re-fixtured from the doctest)."""

import doctest

import morie.fn.km038 as mod


def test_km038_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km038_edge():
    import pytest
    from morie.fn.km038 import kamath_ch2_gpt2_task_conditioning
    with pytest.raises(ValueError):
        kamath_ch2_gpt2_task_conditioning("x", "t", lambda i, t: {"a": 0.5, "b": 0.9})
