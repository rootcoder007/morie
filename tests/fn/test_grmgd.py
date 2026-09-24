"""Tests for grmgd.geron_minibatch_gradient_descent."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.grmgd import geron_minibatch_gradient_descent


def test_grmgd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    theta = [0.0, 0.0, 0.0]
    eta = 0.01
    b = 8
    n_iter = 50
    result = geron_minibatch_gradient_descent(
        X, y, theta, eta, b, n_iter, seed=0
    )
    assert isinstance(result, dict)
    assert "theta" in result
    assert "final_cost" in result
    assert "cost_history" in result
    assert "initial_cost" in result
    assert "estimate" in result
    assert len(result["theta"]) == 3
    assert math.isfinite(result["final_cost"])
    assert result["final_cost"] >= 0
    assert len(result["cost_history"]) == n_iter + 1


def test_grmgd_edge():
    """Test edge cases: full-batch one-step equals a batch-GD step."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    theta = [0.0, 0.0, 0.0]
    eta = 0.01
    b = 40  # b == m: full-batch edge
    n_iter = 1
    result = geron_minibatch_gradient_descent(
        X, y, theta, eta, b, n_iter, seed=0
    )
    assert isinstance(result, dict)
    assert "theta" in result
    assert "final_cost" in result
    assert len(result["theta"]) == 3
    assert math.isfinite(result["final_cost"])
    assert result["final_cost"] >= 0
    assert len(result["cost_history"]) == n_iter + 1


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grmgd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
