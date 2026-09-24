"""Tests for grmpl.geron_max_pooling."""

from morie.fn import _array_core as np

from morie.fn.grmpl import geron_max_pooling


def test_grmpl_basic():
    """Test basic functionality."""
    X = [[1.0, 5.0, 2.0], [3.0, 4.0, 0.0], [9.0, 1.0, 1.0]]
    result = geron_max_pooling(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmpl_edge():
    """Test edge cases."""
    X = [[1.0, 5.0, 2.0], [3.0, 4.0, 0.0], [9.0, 1.0, 1.0]]
    result = geron_max_pooling(X)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grmpl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
