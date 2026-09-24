"""Tests for hmmcel.geron_memory_cell."""

from morie.fn import _array_core as np

from morie.fn.hmmcel import geron_memory_cell


def test_hmmcel_basic():
    """Test basic functionality."""
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_memory_cell(c_prev, x_t, f)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmmcel_edge():
    """Test edge cases."""
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_memory_cell(c_prev, x_t, f)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmmcel as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
