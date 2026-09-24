"""Tests for kmpask.kamath_pass_at_k."""

from morie.fn import _array_core as np

from morie.fn.kmpask import kamath_pass_at_k


def test_kmpask_basic():
    """Test basic functionality."""
    n = 5
    c = 0.5
    k = 5
    result = kamath_pass_at_k(n, c, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmpask_edge():
    """Test edge cases."""
    n = 5
    c = 0.5
    k = 5
    result = kamath_pass_at_k(n, c, k)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmpask as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
