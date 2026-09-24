"""Tests for hmphp.geron_peephole_lstm."""

from morie.fn import _array_core as np

from morie.fn.hmphp import geron_peephole_lstm


def test_hmphp_basic():
    """Test basic functionality."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_peephole_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmphp_edge():
    """Test edge cases."""
    x_t = np.random.default_rng(42).normal(0, 1, 100)
    h_prev = np.random.default_rng(42).normal(0, 1, 100)
    c_prev = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_peephole_lstm(x_t, h_prev, c_prev, weights)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmphp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
