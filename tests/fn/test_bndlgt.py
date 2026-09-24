"""Tests for bndlgt.bound_logistic."""

import math

import pytest
from morie.fn import _array_core as np
from morie.fn.bndlgt import bound_logistic


def test_bndlgt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    y = rng.integers(0, 2, n)
    D = rng.integers(0, 2, n)
    X = rng.integers(0, 3, n)
    result = bound_logistic(y, D, X)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "estimate",
                "p1_lower", "p1_upper", "p0_lower", "p0_upper",
                "n_strata", "n", "method"):
        assert key in result
    assert result["n"] == n
    assert 1 <= result["n_strata"] <= 3
    assert math.isfinite(result["n"])
    assert math.isfinite(result["n_strata"])


def test_bndlgt_edge():
    """Test edge cases with small sample."""
    rng = np.random.default_rng(7)
    n = 40
    y = rng.integers(0, 2, n)
    D = rng.integers(0, 2, n)
    X = rng.integers(0, 2, n)
    result = bound_logistic(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "lower" in result
    assert "upper" in result
    assert "width" in result
    assert result["n"] == n
    assert 1 <= result["n_strata"] <= 2
