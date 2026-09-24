"""Tests for hml1c.geron_one_cycle."""

from morie.fn import _array_core as np

from morie.fn.hml1c import geron_one_cycle


def test_hml1c_basic():
    """Test basic functionality."""
    t = 0.5
    T = 5
    lr_max = 5
    lr_min = 0.5
    result = geron_one_cycle(t, T, lr_max, lr_min)
    assert isinstance(result, dict)
    assert "estimate" in result or "lr" in result


def test_hml1c_edge():
    """Test edge cases."""
    t = 0.5
    T = 5
    lr_max = 5
    lr_min = 0.5
    result = geron_one_cycle(t, T, lr_max, lr_min)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hml1c as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
