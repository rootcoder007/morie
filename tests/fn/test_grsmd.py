"""Tests for grsmd.geron_symbolic_differentiation."""

from morie.fn import _array_core as np

from morie.fn.grsmd import geron_symbolic_differentiation


def test_grsmd_basic():
    """Test basic functionality."""
    # f(x) = x^2 + 2*x + 1, a valid nested-tuple expression tree
    expression = ("+", ("+", ("^", "x", 2), ("*", 2, "x")), 1)
    result = geron_symbolic_differentiation(expression, "x")
    assert isinstance(result, dict)
    assert "derivative" in result
    assert "derivative_str" in result


def test_grsmd_edge():
    """Test edge cases."""
    # Differentiating a constant with respect to x is a valid edge case
    result = geron_symbolic_differentiation(5)
    assert isinstance(result, dict)
    assert "derivative" in result


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grsmd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
