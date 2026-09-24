"""Tests for kmglv.kamath_glove_cost."""

from morie.fn import _array_core as np

from morie.fn.kmglv import kamath_glove_cost


def test_kmglv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    W_tilde = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    b_tilde = np.random.default_rng(42).normal(0, 1, 100)
    x_max = 100
    alpha = 0.05
    result = kamath_glove_cost(X, W, W_tilde, b, b_tilde, x_max, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmglv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    W_tilde = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    b_tilde = np.random.default_rng(42).normal(0, 1, 100)
    x_max = 100
    alpha = 0.05
    result = kamath_glove_cost(X, W, W_tilde, b, b_tilde, x_max, alpha)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmglv as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
