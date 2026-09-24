"""Tests for grlrex.geron_lr_exponential_schedule."""

from morie.fn import _array_core as np

from morie.fn.grlrex import geron_lr_exponential_schedule


def test_grlrex_basic():
    """Test basic functionality."""
    eta0 = 0.1
    gamma = 0.1
    t = 5
    result = geron_lr_exponential_schedule(eta0, gamma, t)
    assert isinstance(result, dict)
    assert "estimate" in result or "eta" in result


def test_grlrex_edge():
    """Test edge cases."""
    eta0 = 0.1
    gamma = 0.1
    t = 5
    result = geron_lr_exponential_schedule(eta0, gamma, t)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grlrex as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
