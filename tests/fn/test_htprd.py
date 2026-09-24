"""Tests for htprd.hyperparameter_tuning_grid."""

import math

from morie.fn import _array_core as np

from morie.fn.htprd import hyperparameter_tuning_grid


def test_htprd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    param_grid = {"lam": [0.01, 0.1, 1.0, 10.0]}
    cv_data = (X, y)
    result = hyperparameter_tuning_grid(param_grid, cv_data)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "best_params" in result
    assert "cv_score" in result
    assert "scores" in result
    assert "grid" in result
    assert "keys" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["cv_score"])
    assert len(result["scores"]) == len(result["grid"])
    assert result["estimate"] == result["cv_score"]


def test_htprd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 20, 2
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    param_grid = {"lam": [0.1, 1.0]}
    cv_data = (X, y)
    result = hyperparameter_tuning_grid(param_grid, cv_data, k=3)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert len(result["scores"]) == 2
    assert result["n"] == 2
    assert result["method"].startswith("argmin")
