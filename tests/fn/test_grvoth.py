"""Tests for grvoth.geron_hard_voting."""

from morie.fn import _array_core as np

from morie.fn.grvoth import geron_hard_voting


def test_grvoth_basic():
    """Test basic functionality."""
    predictions = [[0], [1]]
    result = geron_hard_voting(predictions)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grvoth_edge():
    """Test edge cases."""
    predictions = [[0], [1]]
    result = geron_hard_voting(predictions)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grvoth as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
