"""Tests for hmmbgd.geron_minibatch_gd."""

from morie.fn import _array_core as np
import math

from morie.fn.hmmbgd import geron_minibatch_gd


def test_hmmbgd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    theta = rng.normal(0, 1, 3)
    eta = 0.01
    b = 10
    result = geron_minibatch_gd(X, y, theta, eta, b)
    assert isinstance(result, dict)
    expected_keys = {"theta", "gradient", "full_gradient", "batch_indices",
                     "mse", "estimate", "n", "method"}
    for key in expected_keys:
        assert key in result
    # theta has one entry per feature column
    assert len(result["theta"]) == 3
    # gradient and full_gradient match theta shape
    assert len(result["gradient"]) == 3
    assert len(result["full_gradient"]) == 3
    # batch_indices has length b
    assert len(result["batch_indices"]) == b
    # mse is a finite scalar
    assert math.isfinite(float(result["mse"]))


def test_hmmbgd_edge():
    """Test edge case with b=1 (pure SGD) and several steps."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    theta = rng.normal(0, 1, 3)
    eta = 0.001
    b = 1
    result = geron_minibatch_gd(X, y, theta, eta, b, seed=7, n_steps=3)
    assert isinstance(result, dict)
    expected_keys = {"theta", "gradient", "full_gradient", "batch_indices",
                     "mse", "estimate", "n", "method"}
    for key in expected_keys:
        assert key in result
    # theta still has one entry per feature column after multiple steps
    assert len(result["theta"]) == 3
    # batch of size 1 -> single index
    assert len(result["batch_indices"]) == 1
    # mse is finite after several SGD steps
    assert math.isfinite(float(result["mse"]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmmbgd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
