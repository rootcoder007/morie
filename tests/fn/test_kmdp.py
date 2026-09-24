"""Tests for kmdp.kamath_differential_privacy."""

from morie.fn import _array_core as np

from morie.fn.kmdp import kamath_differential_privacy


def test_kmdp_basic():
    """Test basic functionality."""
    eps = 2.0
    delta = 0.01
    result = kamath_differential_privacy(eps, delta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmdp_edge():
    """Test edge cases."""
    eps = 2.0
    delta = 0.01
    result = kamath_differential_privacy(eps, delta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmdp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
