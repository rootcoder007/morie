"""Tests for autoI.autoint."""

import math

from morie.fn import _array_core as np

from morie.fn.autoI import autoint


def test_autoI_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    M, d = 10, 8
    X = rng.normal(0, 1, (M, d))
    y = int(rng.integers(0, 2))
    K = 2
    result = autoint(X, y, K)
    assert "estimate" in result
    assert "e_res" in result
    assert "attention" in result
    assert "loss" in result
    assert math.isfinite(result["estimate"])
    assert len(result["e_res"]) == M
    assert len(result["attention"]) == M
    assert all(len(row) == M for row in result["attention"])
    assert math.isfinite(result["loss"])


def test_autoI_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    M, d = 5, 4
    X = rng.normal(0, 1, (M, d))
    result = autoint(X)
    assert "estimate" in result
    assert "e_res" in result
    assert "attention" in result
    assert math.isfinite(result["estimate"])
    assert len(result["e_res"]) == M
    assert len(result["attention"]) == M
