"""Tests for kalmS.kalman_smoother."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.kalmS import kalman_smoother


def test_kalmS_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n, d, m = 40, 2, 1
    y = rng.normal(0, 1, (n, m))
    F = rng.normal(0, 1, (d, d))
    H = rng.normal(0, 1, (m, d))
    Q = [[0.1, 0.0], [0.0, 0.1]]
    R = [[0.1]]
    result = kalman_smoother(y, F, H, Q, R)
    assert isinstance(result, dict)
    assert "smoothed" in result
    assert "filtered" in result
    assert "loglik" in result
    assert result["n"] == n
    assert len(result["smoothed"]) == n
    assert len(result["smoothed"][0]) == d
    assert math.isfinite(result["loglik"])


def test_kalmS_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n, d, m = 1, 2, 1
    y = rng.normal(0, 1, (n, m))
    F = rng.normal(0, 1, (d, d))
    H = rng.normal(0, 1, (m, d))
    Q = [[0.1, 0.0], [0.0, 0.1]]
    R = [[0.1]]
    result = kalman_smoother(y, F, H, Q, R)
    assert isinstance(result, dict)
    assert "smoothed" in result
    assert len(result["smoothed"]) == n
    assert len(result["smoothed"][0]) == d
    assert math.isfinite(result["loglik"])
