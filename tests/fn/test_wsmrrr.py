"""Tests for wsmrrr.wasserman_relative_risk (Katz interval)."""

import math

import pytest

from morie.fn.wsmrrr import wasserman_relative_risk


def test_wsmrrr_basic():
    """RR = p1/p0; se(log RR) = sqrt((1-p1)/n11 + (1-p0)/n01); the 95%
    interval is exp(log RR -+ 1.959964 se)."""
    r = wasserman_relative_risk([[18, 42], [9, 51]])
    p1, p0 = 18 / 60, 9 / 60
    se = math.sqrt((1 - p1) / 18 + (1 - p0) / 9)
    assert r["estimate"] == pytest.approx(p1 / p0, rel=1e-15)
    assert r["se"] == pytest.approx(se, rel=1e-15)
    assert r["ci_lower"] == pytest.approx(math.exp(math.log(2.0) - 1.959963984540054 * se), rel=1e-9)
    assert r["ci_upper"] == pytest.approx(math.exp(math.log(2.0) + 1.959963984540054 * se), rel=1e-9)


def test_wsmrrr_edge():
    """Zero events and non-2x2 tables raise."""
    with pytest.raises(ValueError):
        wasserman_relative_risk([[0, 5], [2, 3]])
    with pytest.raises(ValueError):
        wasserman_relative_risk([[1, 2, 3], [4, 5, 6]])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmrrr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
