"""Tests for hmtd.geron_td_learning."""

from morie.fn import _array_core as np

from morie.fn.hmtd import geron_td_learning


def test_hmtd_basic():
    """Test basic functionality."""
    V = [1.0, 2.0]
    s = [0, 1]
    r = [0.5, -1.0]
    s_next = [1, 0]
    result = geron_td_learning(V, s, r, s_next)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmtd_edge():
    """Test edge cases."""
    V = [1.0, 2.0]
    s = [0, 1]
    r = [0.5, -1.0]
    s_next = [1, 0]
    result = geron_td_learning(V, s, r, s_next)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmtd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
