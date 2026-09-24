"""Tests for hmext.geron_extra_trees."""

import math

from morie.fn import _array_core as np

from morie.fn.hmext import geron_extra_trees


def test_hmext_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = geron_extra_trees(
        X, y, n_estimators=10, max_features=2, seed=42, criterion="mse"
    )
    assert isinstance(result, dict)
    assert "predictions" in result
    assert "trees" in result
    assert "train_mse" in result
    assert "n_estimators" in result
    assert result["n_estimators"] == 10
    assert len(result["tree_predictions"]) == 10
    assert len(result["predictions"]) == 40
    assert len(result["trees"]) == 10
    assert math.isfinite(result["train_mse"])
    assert result["train_mse"] >= 0


def test_hmext_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = geron_extra_trees(
        X, y, n_estimators=5, max_features=1, seed=0,
        max_depth=3, min_samples_leaf=2, criterion="mse"
    )
    assert isinstance(result, dict)
    assert "predictions" in result
    assert result["n_estimators"] == 5
    assert len(result["tree_predictions"]) == 5
    assert len(result["predictions"]) == 40
    assert math.isfinite(result["train_mse"])
    assert result["train_mse"] >= 0
