"""Tests for hwmul.holt_winters_mult."""

from morie.fn import _array_core as np

from morie.fn.hwmul import holt_winters_mult


def test_hwmul_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    result = holt_winters_mult(y, period, alpha, beta, gamma)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hwmul_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    gamma = 1.0
    result = holt_winters_mult(y, period, alpha, beta, gamma)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hwmul as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
