"""Tests for hmstk.geron_stacking."""

import math
import pytest
from morie.fn import _array_core as np
from morie.fn.hmstk import geron_stacking


def _mean_base(X_train, y_train, X_test):
    """Base model that always predicts the training mean."""
    y_tr = np.asarray(y_train, dtype=float)
    mean_y = float(np.mean(y_tr))
    X_te = np.asarray(X_test)
    n_test = X_te.shape[0]
    return [mean_y] * n_test


def _linear_base(X_train, y_train, X_test):
    """Base model that fits a simple linear regression on the first feature."""
    X_tr = np.asarray(X_train, dtype=float)
    X_te = np.asarray(X_test, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)

    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)
    if X_te.ndim == 1:
        X_te = X_te.reshape(-1, 1)

    x_tr = [float(X_tr[i][0]) for i in range(X_tr.shape[0])]
    x_te = [float(X_te[i][0]) for i in range(X_te.shape[0])]

    n = len(x_tr)
    mean_x = sum(x_tr) / n
    mean_y = float(np.mean(y_tr))
    num = sum((x_tr[i] - mean_x) * (float(y_tr[i]) - mean_y) for i in range(n))
    den = sum((x_tr[i] - mean_x) ** 2 for i in range(n))
    b = num / den if den != 0 else 0.0
    a = mean_y - b * mean_x

    return [a + b * x_te[i] for i in range(len(x_te))]


def test_hmstk_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)

    base_models = [_mean_base, _linear_base]

    result = geron_stacking(X, y, base_models, k_folds=3)

    assert isinstance(result, dict)
    for key in ["predicted", "meta_features", "oof_mse", "stacked_mse",
                "best_base_mse", "gain", "estimate", "n", "method"]:
        assert key in result, f"Missing key: {key}"

    assert len(result["predicted"]) == 40
    assert result["n"] == 40
    assert result["meta_features"].shape == (40, 2)
    assert math.isfinite(result["stacked_mse"])
    assert math.isfinite(result["best_base_mse"])
    assert math.isfinite(result["gain"])
    assert len(result["oof_mse"]) == 2
    assert isinstance(result["method"], str)
    est = result["estimate"]
    if hasattr(est, "__len__"):
        assert len(est) == 40
    else:
        assert math.isfinite(est)


def test_hmstk_edge():
    """Test edge cases - small sample size with minimum folds."""
    rng = np.random.default_rng(42)
    n = 8
    X = rng.normal(0, 1, (n, 2))
    y = rng.normal(0, 1, n)

    base_models = [_mean_base, _linear_base]

    result = geron_stacking(X, y, base_models, k_folds=2)

    assert isinstance(result, dict)
    assert "stacked_mse" in result
    assert "method" in result
    assert result["n"] == n
    assert len(result["predicted"]) == n
    assert math.isfinite(result["stacked_mse"])
    assert isinstance(result["method"], str)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmstk as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
