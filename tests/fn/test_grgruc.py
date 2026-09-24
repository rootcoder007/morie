"""Tests for grgruc.geron_gru_cell."""

from morie.fn import _array_core as np

from morie.fn.grgruc import geron_gru_cell


def test_grgruc_basic():
    """Test basic functionality."""
    x_t = [1.0]
    h_prev = [0.6]
    Wz = [[0.0, 1.0]]
    Wr = [[0.0, 2.0]]
    W = [[1.0, 0.0]]
    result = geron_gru_cell(x_t, h_prev, Wz, Wr, W)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grgruc_edge():
    """Test edge cases."""
    x_t = [1.0]
    h_prev = [0.6]
    Wz = [[0.0, 1.0]]
    Wr = [[0.0, 2.0]]
    W = [[1.0, 0.0]]
    result = geron_gru_cell(x_t, h_prev, Wz, Wr, W)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grgruc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
