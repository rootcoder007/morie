"""Tests for grovo.geron_one_vs_one."""

from morie.fn import _array_core as np

from morie.fn.grovo import geron_one_vs_one


def test_grovo_basic():
    """Test basic functionality."""
    X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    y = [0, 0, 1, 1, 2, 2]
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grovo_edge():
    """Test edge cases."""
    X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    y = [0, 0, 1, 1, 2, 2]
    result = geron_one_vs_one(X, y)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grovo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
