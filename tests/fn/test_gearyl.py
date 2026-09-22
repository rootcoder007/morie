"""Tests for gearyl.local_gearys_c."""

from morie.fn import _array_core as np

from morie.fn.gearyl import local_gearys_c


def test_gearyl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(0, 1, n)
    W = rng.normal(0, 1, (n, n))
    result = local_gearys_c(x, W)
    assert isinstance(result, dict)
    assert "local" in result
    assert "global_c" in result
    assert "z" in result
    assert "n" in result

    # Independent recomputation of the documented formula.
    mu = sum(x) / n
    # sample standard deviation (ddof=1) per the function's reference impl
    var = sum((v - mu) ** 2 for v in x) / (n - 1)
    sd = var ** 0.5
    z = [(v - mu) / sd for v in x]
    expected_local = [
        sum(W[i][j] * (z[i] - z[j]) ** 2 for j in range(n))
        for i in range(n)
    ]
    s0 = sum(sum(row) for row in W)
    expected_global = sum(expected_local) / (2.0 * s0)

    assert len(result["local"]) == n
    assert result["n"] == n
    for got, exp in zip(result["local"], expected_local):
        assert abs(got - exp) < 1e-9
    assert abs(result["global_c"] - expected_global) < 1e-9


def test_gearyl_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 50
    x = rng.normal(0, 1, n)
    W = rng.normal(0, 1, (n, n))
    result = local_gearys_c(x, W)
    assert isinstance(result, dict)
    assert "local" in result
    assert len(result["local"]) == n
