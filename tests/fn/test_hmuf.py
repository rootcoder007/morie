"""Tests for hmuf.geron_underfitting."""

import math

import pytest

from morie.fn.hmuf import geron_underfitting


def test_hmuf_basic():
    """Test basic functionality."""
    # High training error with small gap -> underfitting (bias)
    result = geron_underfitting(0.40, threshold=0.10, val_err=0.41)
    assert "diagnosis" in result
    assert result["diagnosis"] == "underfitting"
    assert bool(result["underfitting"]) is True
    assert "gap" in result
    gap = float(result["gap"])
    assert math.isfinite(gap)
    assert abs(gap - 0.01) < 1e-9


def test_hmuf_edge():
    """Test edge cases."""
    # Empty training-error input is rejected by the function.
    with pytest.raises(ValueError):
        geron_underfitting([], threshold=0.10)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmuf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
