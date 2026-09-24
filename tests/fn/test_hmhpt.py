"""Tests for hmhpt.geron_hyperparameter_tuning."""

import math

from morie.fn import _array_core as np

from morie.fn.hmhpt import geron_hyperparameter_tuning


def test_hmhpt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    param_grid = {"alpha": [0.0, 0.1, 1.0, 10.0]}
    result = geron_hyperparameter_tuning(param_grid, X, y, K=3, seed=0)
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert "n_candidates" in result
    assert "n_fits" in result
    assert "method" in result
    assert math.isfinite(result["best_score"])
    assert result["n_candidates"] == 4
    assert result["n_fits"] == 4 * 3
    assert isinstance(result["method"], str)
    assert len(result["method"]) > 0


def test_hmhpt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    param_grid = {"alpha": [0.0, 0.1, 1.0, 10.0]}
    result = geron_hyperparameter_tuning(
        param_grid, X, y, search="random", n_iter=3, K=2, seed=0
    )
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert result["n_candidates"] == 3
    assert result["n_fits"] == 3 * 2
    assert isinstance(result["method"], str)
    assert len(result["method"]) > 0
    assert math.isfinite(result["best_score"])
