"""Tests for jowfv.joseph_walk_forward_validation."""

import math

from morie.fn import _array_core as np

from morie.fn.jowfv import joseph_walk_forward_validation


def test_jowfv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 100)
    model = rng.normal(0, 1, 100)
    T_start = 50
    testsize = 10
    result = joseph_walk_forward_validation(y, model, T_start, testsize)
    assert isinstance(result, dict)
    assert "rmse" in result
    assert "sd" in result
    assert "best" in result
    assert "worst" in result
    assert math.isfinite(result["rmse"])
    assert math.isfinite(result["sd"])
    assert math.isfinite(result["best"])
    assert math.isfinite(result["worst"])
    assert result["sd"] >= 0
    assert result["best"] <= result["rmse"] <= result["worst"]


def test_jowfv_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 100)
    model = rng.normal(0, 1, 100)
    T_start = 80
    testsize = 5
    result = joseph_walk_forward_validation(y, model, T_start, testsize)
    assert isinstance(result, dict)
    assert "rmse" in result
    assert "sd" in result
    assert "best" in result
    assert "worst" in result
    assert math.isfinite(result["rmse"])
    assert math.isfinite(result["sd"])
    assert math.isfinite(result["best"])
    assert math.isfinite(result["worst"])
    assert result["sd"] >= 0
    assert result["best"] <= result["rmse"] <= result["worst"]
