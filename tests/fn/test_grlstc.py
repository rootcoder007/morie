"""Tests for grlstc.geron_lstm_cell."""

from morie.fn import _array_core as np

from morie.fn.grlstc import geron_lstm_cell


def test_grlstc_basic():
    """Test basic functionality."""
    x_t = [2.0]
    h_prev = [0.0]
    c_prev = [0.5]
    Wf = [[0.0, 1.0]]
    Wi = [[0.0, 1.0]]
    Wg = [[0.0, 1.0]]
    Wo = [[0.0, 1.0]]
    bf = 0.0
    bi = 0.0
    bg = 0.0
    bo = 0.0
    result = geron_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlstc_edge():
    """Test edge cases."""
    x_t = [2.0]
    h_prev = [0.0]
    c_prev = [0.5]
    Wf = [[0.0, 1.0]]
    Wi = [[0.0, 1.0]]
    Wg = [[0.0, 1.0]]
    Wo = [[0.0, 1.0]]
    bf = 0.0
    bi = 0.0
    bg = 0.0
    bo = 0.0
    result = geron_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grlstc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
