"""Tests for grnag.geron_nesterov_accelerated_gradient."""

from morie.fn import _array_core as np

from morie.fn.grnag import geron_nesterov_accelerated_gradient


def test_grnag_basic():
    """Test basic functionality."""
    theta = 0.0
    grad_fn = lambda v: v
    v = np.random.default_rng(44).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_nesterov_accelerated_gradient(theta, grad_fn, v, eta, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grnag_edge():
    """Test edge cases."""
    theta = 0.0
    grad_fn = lambda v: v
    v = np.random.default_rng(44).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_nesterov_accelerated_gradient(theta, grad_fn, v, eta, beta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grnag as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
