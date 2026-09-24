"""Tests for grnag.geron_nesterov_accelerated_gradient."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.grnag import geron_nesterov_accelerated_gradient


def test_grnag_basic():
    """Test basic functionality with a quadratic loss J(theta) = ||theta||**2."""
    rng = np.random.default_rng(0)
    p = 3
    theta = rng.normal(0, 1, p)
    v = rng.normal(0, 1, p)
    eta = 0.1
    beta = 0.8
    # gradient of theta^T theta is 2 theta
    grad_fn = lambda t: [2.0 * x for x in t]
    result = geron_nesterov_accelerated_gradient(theta, grad_fn, v, eta, beta)
    assert isinstance(result, dict)
    for key in (
        "theta_new",
        "v_new",
        "lookahead",
        "gradient",
        "path",
        "step",
        "estimate",
        "n",
        "method",
    ):
        assert key in result
    assert len(result["theta_new"]) == p
    assert len(result["v_new"]) == p
    assert len(result["lookahead"]) == p
    assert len(result["gradient"]) == p
    assert all(math.isfinite(float(x)) for x in result["theta_new"])
    assert all(math.isfinite(float(x)) for x in result["v_new"])
    assert all(math.isfinite(float(x)) for x in result["lookahead"])
    assert all(math.isfinite(float(x)) for x in result["gradient"])


def test_grnag_edge():
    """Test that invalid inputs raise ValueError per the function's contract."""
    grad_fn = lambda t: [2.0 * x for x in t]
    # Empty theta
    with pytest.raises(ValueError):
        geron_nesterov_accelerated_gradient([], grad_fn, [], eta=0.1)
    # v and theta have different shapes
    with pytest.raises(ValueError):
        geron_nesterov_accelerated_gradient([1.0, 2.0], grad_fn, [1.0], eta=0.1)
    # Non-positive eta
    with pytest.raises(ValueError):
        geron_nesterov_accelerated_gradient([1.0], grad_fn, [0.0], eta=0.0)
    # beta >= 1 violates the [0, 1) contract
    with pytest.raises(ValueError):
        geron_nesterov_accelerated_gradient(
            [1.0], grad_fn, [0.0], eta=0.1, beta=1.0
        )
    # grad_fn must be callable
    with pytest.raises(ValueError):
        geron_nesterov_accelerated_gradient([1.0], 42, [0.0], eta=0.1)


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
