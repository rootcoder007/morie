"""Tests for grlaso.geron_lasso_cost."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.grlaso import geron_lasso_cost


def test_grlaso_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    # Build X with a bias column prepended so shape is (n, p).
    features = rng.normal(0, 1, (n, p - 1))
    X = [[1.0] + list(row) for row in features]
    # Generate y from a known linear model with no noise (MSE is computable).
    w = [0.5, -1.0, 2.0]
    y = [sum(w[j] * X[i][j] for j in range(p)) for i in range(n)]

    # Perfect fit: theta equals the true weights used to generate y.
    theta = list(w)
    alpha = 0.05

    result = geron_lasso_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
    # Keys promised by the docstring.
    for key in ("cost", "mse", "l1_penalty", "l1_norm", "n_zero",
                "estimate", "n", "method"):
        assert key in result
    # Scalar outputs must be finite.
    assert math.isfinite(result["cost"])
    assert math.isfinite(result["mse"])
    assert math.isfinite(result["l1_penalty"])
    assert math.isfinite(result["l1_norm"])
    # cost = mse + alpha * l1_norm (penalize_intercept=False, so theta[0] is excluded).
    assert abs(result["cost"] - (result["mse"] + result["l1_penalty"])) < 1e-10
    assert abs(result["l1_penalty"] - alpha * result["l1_norm"]) < 1e-10
    expected_l1 = abs(theta[1]) + abs(theta[2])
    assert abs(result["l1_norm"] - expected_l1) < 1e-10
    # Perfect fit: mse is zero, so cost equals the penalty.
    assert result["mse"] == 0.0


def test_grlaso_edge():
    """Test edge cases: negative alpha is documented as invalid."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    features = rng.normal(0, 1, (n, p - 1))
    X = [[1.0] + list(row) for row in features]
    y = rng.normal(0, 1, n)
    theta = [0.1, 0.2, 0.3]

    with pytest.raises(ValueError):
        geron_lasso_cost(X, y, theta, alpha=-0.1)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grlaso as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
