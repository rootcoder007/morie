"""Tests for otmd.ot_mahalanobis_distance_ot."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.otmd import ot_mahalanobis_distance_ot


def test_otmd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    d = 3
    n = 40
    m = 50
    X = rng.normal(0, 1, (n, d))
    Y = rng.normal(0, 1, (m, d))
    # Build a diagonal covariance matrix (symmetric positive definite)
    var = [rng.uniform(0.5, 2.0) for _ in range(d)]
    Sigma = [[var[i] if i == j else 0.0 for j in range(d)] for i in range(d)]

    result = ot_mahalanobis_distance_ot(X, Y, Sigma)

    # Expected keys from the RichResult
    assert "C" in result
    assert "cost" in result
    assert "n" in result
    assert "m" in result
    assert "d" in result
    assert "method" in result

    assert result["n"] == n
    assert result["m"] == m
    assert result["d"] == d

    C = result["C"]
    assert len(C) == n
    for row in C:
        assert len(row) == m

    assert math.isfinite(result["cost"])
    assert result["cost"] >= 0.0


def test_otmd_edge():
    """Test edge case with minimal sizes."""
    rng = np.random.default_rng(1)
    d = 1
    n = 1
    m = 1
    X = rng.normal(0, 1, (n, d))
    Y = rng.normal(0, 1, (m, d))
    Sigma = [[1.0]]  # 1x1 identity covariance

    result = ot_mahalanobis_distance_ot(X, Y, Sigma)

    assert "C" in result
    assert "cost" in result
    assert result["n"] == n
    assert result["m"] == m
    assert result["d"] == d

    assert len(result["C"]) == n
    assert len(result["C"][0]) == m
    assert math.isfinite(result["cost"])
    assert result["cost"] >= 0.0
