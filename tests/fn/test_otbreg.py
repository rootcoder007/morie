"""Tests for otbreg.ot_bregman_proj."""

import math

from morie.fn import _array_core as np

from morie.fn.otbreg import ot_bregman_proj


def test_otbreg_basic():
    """Test basic functionality."""
    n, m = 10, 10
    rng = np.random.default_rng(43)
    # Entrywise positive Gibbs kernel
    K = rng.uniform(0.1, 2.0, (n, m))
    # Non-negative marginals of matching length
    a = rng.uniform(0.1, 1.0, n)
    b = rng.uniform(0.1, 1.0, m)
    result = ot_bregman_proj(K, a, b, max_iter=100)
    assert isinstance(result, dict)
    assert "T" in result
    assert "iters" in result
    assert "row_err" in result
    assert "col_err" in result
    assert "n" in result
    assert "m" in result
    assert result["iters"] == 100
    assert result["n"] == n
    assert result["m"] == m
    T = result["T"]
    assert len(T) == n
    assert len(T[0]) == m
    assert math.isfinite(result["row_err"])
    assert math.isfinite(result["col_err"])


def test_otbreg_edge():
    """Test edge cases."""
    n, m = 5, 5
    rng = np.random.default_rng(44)
    K = rng.uniform(0.5, 1.5, (n, m))
    a = rng.uniform(0.1, 1.0, n)
    b = rng.uniform(0.1, 1.0, m)
    # Default max_iter=200
    result = ot_bregman_proj(K, a, b)
    assert isinstance(result, dict)
    assert result["iters"] == 200
    assert result["n"] == n
    assert result["m"] == m
    assert math.isfinite(result["row_err"])
    assert math.isfinite(result["col_err"])
    T = result["T"]
    assert len(T) == n
    assert len(T[0]) == m
