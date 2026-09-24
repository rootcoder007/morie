"""Tests for grf1.geron_f1_score."""

from morie.fn import _array_core as np

from morie.fn.grf1 import geron_f1_score


def test_grf1_basic():
    """Test basic functionality."""
    y_true = [1, 1, 1, 0]
    y_pred = [1, 0, 0, 0]
    result = geron_f1_score(y_true, y_pred)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grf1_edge():
    """Test edge cases."""
    y_true = [1, 1, 1, 0]
    y_pred = [1, 0, 0, 0]
    result = geron_f1_score(y_true, y_pred)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grf1 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
