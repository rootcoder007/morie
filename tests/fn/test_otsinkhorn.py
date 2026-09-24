"""Tests for otsinkhorn.ot_sinkhorn."""

import math

from morie.fn import _array_core as np

from morie.fn.otsinkhorn import ot_sinkhorn


def test_otsinkhorn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 5, 7
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, m)
    C = rng.uniform(0, 1, (n, m))
    epsilon = 0.1
    max_iter = 50
    result = ot_sinkhorn(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "T" in result
    assert "u" in result
    assert "v" in result
    assert "iters" in result
    assert "marginal_error" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0
    assert result["iters"] == 50
    assert result["marginal_error"] >= 0


def test_otsinkhorn_edge():
    """Test edge cases with small valid input."""
    a = [1.0, 2.0, 3.0]
    b = [1.0, 1.0]
    C = [[0.5, 1.5], [1.0, 2.0], [0.8, 0.3]]
    epsilon = 0.05
    max_iter = 10
    result = ot_sinkhorn(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "T" in result
    assert "u" in result
    assert "v" in result
    assert "iters" in result
    assert "marginal_error" in result
    assert "method" in result
    assert result["iters"] == 10
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0
