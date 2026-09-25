"""Tests for kmyarn.kamath_yarn_context_extrapolation (NTK-aware RoPE scaling)."""

import pytest

from morie.fn.kmyarn import kamath_yarn_context_extrapolation


def test_kmyarn_basic():
    """theta_i = base^(-2i/d) and theta_i' = theta_i / s^(2i/d) for
    i = 0 .. d/2 - 1, recomputed for base 10000, d = 8, s = 4: the
    highest frequency is untouched, the lowest divided by s^(6/8)."""
    result = kamath_yarn_context_extrapolation(10000.0, 4.0, 8)
    assert isinstance(result, dict)
    th = [10000.0 ** (-2 * i / 8) for i in range(4)]
    assert result["theta"] == pytest.approx(th, rel=1e-14)
    assert result["theta_new"] == pytest.approx(
        [t / 4.0 ** (2 * i / 8) for i, t in enumerate(th)], rel=1e-14)
    assert result["theta_new"][0] == result["theta"][0]
    assert result["effective_context_multiplier"] == 4.0


def test_kmyarn_edge():
    """A scalar base must exceed 1; s = 1 changes nothing."""
    with pytest.raises(ValueError):
        kamath_yarn_context_extrapolation(0.5, 2.0, 4)
    r = kamath_yarn_context_extrapolation(500.0, 1.0, 6)
    assert r["theta_new"] == pytest.approx(r["theta"], rel=1e-15)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.kmyarn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
