"""Tests for grepl.geron_epsilon_greedy."""

from morie.fn import _array_core as np

from morie.fn.grepl import geron_epsilon_greedy


def test_grepl_basic():
    """Test basic functionality."""
    Q_s = [1.0, 7.0, 3.0, 2.0]
    eps = 0.4
    result = geron_epsilon_greedy(Q_s, eps)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grepl_edge():
    """Test edge cases."""
    Q_s = [1.0, 7.0, 3.0, 2.0]
    eps = 0.4
    result = geron_epsilon_greedy(Q_s, eps)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grepl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
