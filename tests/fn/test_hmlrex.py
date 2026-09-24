"""Tests for hmlrex.geron_lr_exponential."""

from morie.fn import _array_core as np

from morie.fn.hmlrex import geron_lr_exponential


def test_hmlrex_basic():
    """Test basic functionality."""
    eta0 = 0.5
    decay = 0.5
    t = 0.5
    result = geron_lr_exponential(eta0, decay, t)
    assert isinstance(result, dict)
    assert "estimate" in result or "eta" in result


def test_hmlrex_edge():
    """Test edge cases."""
    eta0 = 0.5
    decay = 0.5
    t = 0.5
    result = geron_lr_exponential(eta0, decay, t)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmlrex as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
