"""Tests for groob.geron_oob_error."""

from morie.fn import _array_core as np

from morie.fn.groob import geron_oob_error


def test_groob_basic():
    """Test basic functionality."""
    y = [0.0, 1.0]
    predictions = [[9.0, 1.0], [0.0, 9.0]]
    in_bag = [[True, False], [False, True]]
    result = geron_oob_error(y, predictions, in_bag)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_groob_edge():
    """Test edge cases."""
    y = [0.0, 1.0]
    predictions = [[9.0, 1.0], [0.0, 9.0]]
    in_bag = [[True, False], [False, True]]
    result = geron_oob_error(y, predictions, in_bag)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.groob as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
