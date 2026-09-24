"""Tests for wsmhfd.wasserman_hoeffding."""

from morie.fn import _array_core as np

from morie.fn.wsmhfd import wasserman_hoeffding


def test_wsmhfd_basic():
    """Test basic functionality."""
    n = 5
    t = 0.5
    a = 0.5
    b = 5
    result = wasserman_hoeffding(n, t, a, b)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmhfd_edge():
    """Test edge cases."""
    n = 5
    t = 0.5
    a = 0.5
    b = 5
    result = wasserman_hoeffding(n, t, a, b)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmhfd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
