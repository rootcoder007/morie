"""Tests for wsmodd.wasserman_odds_ratio (Woolf interval)."""

import math

import pytest

from morie.fn.wsmodd import wasserman_odds_ratio


def test_wsmodd_basic():
    """OR = n11 n00 / (n10 n01); se(log OR) = sqrt(sum 1/n); the 95%
    interval is exp(log OR -+ 1.959964 se)."""
    t = [[12, 7], [5, 20]]
    r = wasserman_odds_ratio(t)
    orr = 12 * 20 / (7 * 5)
    se = math.sqrt(1 / 12 + 1 / 7 + 1 / 5 + 1 / 20)
    assert r["estimate"] == pytest.approx(orr, rel=1e-15)
    assert r["log_or"] == pytest.approx(math.log(orr), rel=1e-15)
    assert r["se"] == pytest.approx(se, rel=1e-15)
    assert r["ci_lower"] == pytest.approx(math.exp(math.log(orr) - 1.959963984540054 * se), rel=1e-9)
    assert r["ci_upper"] == pytest.approx(math.exp(math.log(orr) + 1.959963984540054 * se), rel=1e-9)
    assert r["n"] == 44


def test_wsmodd_edge():
    """A zero cell and a non-2x2 table raise."""
    with pytest.raises(ValueError):
        wasserman_odds_ratio([[5, 0], [3, 2]])
    with pytest.raises(ValueError):
        wasserman_odds_ratio([[1, 2, 3], [4, 5, 6]])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmodd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
