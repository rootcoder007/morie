"""Tests for hmdthv.geron_tree_high_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.hmdthv import geron_tree_high_variance


def test_hmdthv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = geron_tree_high_variance(X, y, criterion="mse")
    assert isinstance(result, dict)
    assert "variance" in result
    assert "bias2" in result
    assert "structural_instability" in result
    assert "root_splits" in result
    assert "estimate" in result
    assert math.isfinite(result["variance"])
    assert math.isfinite(result["bias2"])
    assert 0.0 <= result["structural_instability"] <= 1.0
    assert isinstance(result["root_splits"], list)


def test_hmdthv_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    result = geron_tree_high_variance(X, y, n_resamples=5, seed=0, max_depth=3, criterion="mse")
    assert isinstance(result, dict)
    assert "variance" in result
    assert "bias2" in result
    assert "structural_instability" in result
    assert isinstance(result["root_splits"], list)
    assert len(result["root_splits"]) == 5
    assert math.isfinite(result["variance"])
    assert math.isfinite(result["bias2"])
    assert 0.0 <= result["structural_instability"] <= 1.0
