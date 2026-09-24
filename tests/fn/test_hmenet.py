"""Tests for hmenet.geron_elastic_net."""

import math

from morie.fn import _array_core as np

from morie.fn.hmenet import geron_elastic_net


def test_hmenet_basic():
    """Test basic functionality with random data and r=0.5."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_theta = np.random.default_rng(44)
    n, p = 40, 3
    X = rng_x.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    # With fit_intercept=True (default), theta has length p+1
    theta = rng_theta.normal(0, 1, p + 1)
    alpha = 0.05
    r = 0.5
    result = geron_elastic_net(X, y, theta, alpha, r)
    assert isinstance(result, dict)
    assert "cost" in result
    assert "mse" in result
    assert "gradient" in result
    assert math.isfinite(result["cost"])
    assert math.isfinite(result["mse"])
    assert result["gradient"].size == p + 1


def test_hmenet_edge():
    """Test edge case at the lasso endpoint r=1 (no L2 penalty)."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_theta = np.random.default_rng(44)
    n, p = 40, 3
    X = rng_x.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    theta = rng_theta.normal(0, 1, p + 1)
    alpha = 0.05
    r = 1.0
    result = geron_elastic_net(X, y, theta, alpha, r)
    assert isinstance(result, dict)
    assert "l1_penalty" in result
    assert "l2_penalty" in result
    assert result["l2_penalty"] == 0.0  # r=1 collapses the L2 term
    assert math.isfinite(result["cost"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmenet as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
