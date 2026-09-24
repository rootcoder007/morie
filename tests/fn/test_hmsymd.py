"""Tests for hmsymd.geron_symbolic_diff."""

from morie.fn import _array_core as np

from morie.fn.hmsymd import geron_symbolic_diff


def test_hmsymd_basic():
    """Test basic functionality."""
    result = geron_symbolic_diff("x^2", "x")
    assert isinstance(result, dict)
    assert "derivative" in result
    assert "tree" in result
    assert "expression" in result
    assert "value" in result
    assert "numeric_check" in result
    assert "error" in result
    assert "nodes" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result


def test_hmsymd_edge():
    """Test edge cases."""
    # Differentiating a constant (no occurrence of x) yields zero
    result = geron_symbolic_diff("y^3", "x")
    assert isinstance(result, dict)
    assert "derivative" in result
    assert result["derivative"] == "0"


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmsymd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
