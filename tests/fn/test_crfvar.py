"""Tests for crfvar.causal_forest_variance."""

from morie.fn import _array_core as np

import math
import pytest

from morie.fn.crfvar import causal_forest_variance
from morie.fn.cfst import causal_forest


def _build_forest(n=120, p=2, n_trees=20, seed=0):
    """Create a small synthetic dataset and fit a causal forest."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    T = rng.integers(0, 2, n).astype(float)
    Y = T * X[:, 0] + rng.normal(0, 0.3, n)
    forest = causal_forest(Y, T, X, n_trees=n_trees, seed=seed + 1)
    return forest, X


def test_crfvar_basic():
    """Basic functionality: variance, se, predictions have correct shape."""
    forest, X = _build_forest(n=120, p=2, n_trees=20, seed=0)
    X_test = X[:5]
    result = causal_forest_variance(forest, X_test)

    expected_keys = {
        "variance", "se", "ci_lower", "ci_upper", "predictions",
        "variance_raw", "bias_raw", "correction_share", "n_trees", "reliable",
    }
    assert expected_keys.issubset(set(result.keys()))

    # predictions and se have length equal to number of test points
    assert len(result["predictions"]) == len(X_test)
    assert len(result["se"]) == len(X_test)

    # standard errors must be finite and positive
    for s in result["se"]:
        assert math.isfinite(s) and s > 0

    # number of trees matches the forest
    assert result["n_trees"] == 20

    # reliable flag is boolean
    assert result["reliable"] in (True, False)


def test_crfvar_edge():
    """Edge case: omitting X_test evaluates at the training rows."""
    forest, X = _build_forest(n=120, p=2, n_trees=20, seed=0)
    result = causal_forest_variance(forest)  # X_test defaults to training rows

    # the predictions should have length equal to the number of training rows
    assert len(result["predictions"]) == 120
    assert len(result["se"]) == 120

    # standard errors are finite and positive
    for s in result["se"]:
        assert math.isfinite(s) and s > 0

    # all expected keys are present
    expected_keys = {
        "variance", "se", "ci_lower", "ci_upper", "predictions",
        "variance_raw", "bias_raw", "correction_share", "n_trees", "reliable",
    }
    assert expected_keys.issubset(set(result.keys()))
