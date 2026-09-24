"""Tests for hmresn.geron_resnet."""

import math

from morie.fn import _array_core as np

import pytest

from morie.fn.hmresn import geron_resnet


def test_hmresn_basic():
    """Test basic functionality with a callable F returning the same shape."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10)
    # F must be a callable; it returns the input scaled (same shape).
    F = lambda a: np.asarray(a, dtype=float) * 0.5
    result = geron_resnet(x, F)
    assert isinstance(result, dict)
    # Keys named in the return / docstring
    for key in ("y", "skip", "residual", "residual_fraction", "method"):
        assert key in result
    y = np.asarray(result["y"])
    skip = np.asarray(result["skip"])
    assert y.shape == skip.shape
    # F produces finite values and the block output is finite.
    assert all(math.isfinite(float(v)) for v in y)


def test_hmresn_edge():
    """Test edge case: dead residual branch (F returns zeros) leaves x unchanged."""
    x = [1.0, 2.0, 3.0]
    # Dead branch: F returns zeros of the same shape, so y == x.
    F = lambda a: np.zeros(np.asarray(a).shape)
    result = geron_resnet(x, F)
    assert isinstance(result, dict)
    assert "y" in result
    assert "residual_fraction" in result
    y = [float(v) for v in result["y"]]
    assert y == [1.0, 2.0, 3.0]
    # ||F(x)|| == 0  ==>  residual fraction is 0
    assert float(result["residual_fraction"]) == 0.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmresn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
