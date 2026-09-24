"""Tests for grelas.geron_elastic_net_cost."""

import math

from morie.fn import _array_core as np

from morie.fn.grelas import geron_elastic_net_cost


def test_grelas_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    theta = np.zeros(3)
    alpha = 0.05
    r = 0.5
    result = geron_elastic_net_cost(X, y, theta, alpha, r)
    assert isinstance(result, dict)
    for key in ("cost", "mse", "l1_penalty", "l2_penalty",
                "l1_norm", "l2_norm_sq", "estimate", "n", "method"):
        assert key in result
    assert math.isfinite(result["cost"])
    assert math.isfinite(result["mse"])
    assert result["l2_penalty"] >= 0.0
    assert result["cost"] >= result["mse"]


def test_grelas_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    theta = np.ones(3)
    alpha = 0.1
    r = 1.0
    result = geron_elastic_net_cost(X, y, theta, alpha, r)
    assert isinstance(result, dict)
    assert math.isfinite(result["cost"])
    # At r = 1 the L2 arm is switched off, so elastic net collapses to lasso
    assert result["l2_penalty"] == 0.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grelas as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
