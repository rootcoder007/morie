"""Tests for grsgd.geron_stochastic_gradient_descent."""

from morie.fn import _array_core as np

from morie.fn.grsgd import geron_stochastic_gradient_descent


def test_grsgd_basic():
    """Test basic functionality."""
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    y = [4.0, 7.0, 10.0]
    theta = [0.0, 0.0]
    eta = 0.05
    n_iter = 150
    result = geron_stochastic_gradient_descent(X, y, theta, eta, n_iter)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsgd_edge():
    """Test edge cases."""
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    y = [4.0, 7.0, 10.0]
    theta = [0.0, 0.0]
    eta = 0.05
    n_iter = 150
    result = geron_stochastic_gradient_descent(X, y, theta, eta, n_iter)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grsgd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
