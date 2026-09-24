"""Tests for grlrnc.geron_learning_curves."""

from morie.fn import _array_core as np

from morie.fn.grlrnc import geron_learning_curves


def test_grlrnc_basic():
    """Test basic functionality."""
    X = [[1.0, float(i)] for i in range(12)]
    y = [2.0 * i + 1 for i in range(12)]
    result = geron_learning_curves(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlrnc_edge():
    """Test edge cases."""
    X = [[1.0, float(i)] for i in range(12)]
    y = [2.0 * i + 1 for i in range(12)]
    result = geron_learning_curves(X, y)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grlrnc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
