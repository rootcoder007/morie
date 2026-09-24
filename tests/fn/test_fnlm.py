"""Tests for fnlm.function_on_function."""

import math

from morie.fn import _array_core as np

from morie.fn.fnlm import function_on_function


def test_fnlm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    N, ns, nt, K1, K2 = 40, 10, 8, 3, 3
    X = rng.normal(0, 1, (N, ns))
    Y = rng.normal(0, 1, (N, nt))
    basis_X = rng.normal(0, 1, (ns, K1))
    basis_Y = rng.normal(0, 1, (nt, K2))
    result = function_on_function(X, Y, basis_X, basis_Y)
    assert isinstance(result, dict)
    for key in ("estimate", "B", "beta", "Z", "fitted",
                "residual", "sse", "ssy", "r2"):
        assert key in result
    assert len(result["B"]) == K1
    assert all(len(row) == K2 for row in result["B"])
    assert len(result["beta"]) == ns
    assert all(len(row) == nt for row in result["beta"])
    assert len(result["Z"]) == N
    assert all(len(row) == K1 for row in result["Z"])
    assert len(result["fitted"]) == N
    assert all(len(row) == nt for row in result["fitted"])
    assert len(result["residual"]) == N
    assert all(len(row) == nt for row in result["residual"])
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["sse"])
    assert math.isfinite(result["ssy"])
    assert math.isfinite(result["r2"])


def test_fnlm_edge():
    """Test edge cases with custom grids and small dimensions."""
    rng = np.random.default_rng(7)
    N, ns, nt, K1, K2 = 30, 5, 5, 2, 2
    X = rng.normal(0, 1, (N, ns))
    Y = rng.normal(0, 1, (N, nt))
    basis_X = rng.normal(0, 1, (ns, K1))
    basis_Y = rng.normal(0, 1, (nt, K2))
    s = np.linspace(0, 1, ns)
    t = np.linspace(0, 1, nt)
    result = function_on_function(X, Y, basis_X, basis_Y, s=s, t=t)
    assert isinstance(result, dict)
    assert "B" in result
    assert "r2" in result
    assert len(result["B"]) == K1
    assert all(len(row) == K2 for row in result["B"])
    assert len(result["beta"]) == ns
    assert all(len(row) == nt for row in result["beta"])
    assert math.isfinite(result["sse"])
    assert math.isfinite(result["r2"])
