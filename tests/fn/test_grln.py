"""Tests for grln.geron_layer_normalization."""

from morie.fn import _array_core as np

from morie.fn.grln import geron_layer_normalization


def test_grln_basic():
    """Test basic functionality."""
    X = [[0.5, -1.5, 2.0], [3.0, 0.0, -1.0]]
    result = geron_layer_normalization(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grln_edge():
    """Test edge cases."""
    X = [[0.5, -1.5, 2.0], [3.0, 0.0, -1.0]]
    result = geron_layer_normalization(X)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grln as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
