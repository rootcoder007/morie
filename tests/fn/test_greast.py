"""Tests for greast.geron_early_stopping."""

from morie.fn import _array_core as np

from morie.fn.greast import geron_early_stopping


def test_greast_basic():
    """Test basic functionality."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    X_val = np.random.default_rng(42).normal(0, 1, 100)
    y_val = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_greast_edge():
    """Test edge cases."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    X_val = np.random.default_rng(42).normal(0, 1, 100)
    y_val = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.greast as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
