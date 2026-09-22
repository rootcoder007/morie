"""Tests for gb521.gibbons_quantile_ci."""

from morie.fn import _array_core as np
import math

from morie.fn.gb521 import gibbons_quantile_ci


def test_gb521_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 0.5
    r = 25
    s = 75
    result = gibbons_quantile_ci(x, p, r, s)
    assert isinstance(result, dict)
    # Documented keys from the function
    assert "lower" in result
    assert "upper" in result
    assert "coverage" in result
    assert "alpha" in result
    assert "r" in result
    assert "s" in result
    assert "n" in result
    assert "p" in result
    assert "method" in result

    # Sanity: lower <= upper (order stats r-1 and s-1 of sorted sample)
    xs = sorted(float(v) for v in x)
    assert result["lower"] == xs[r - 1]
    assert result["upper"] == xs[s - 1]

    # Coverage computed independently from the documented formula:
    # sum_{i=r}^{s-1} C(n, i) p^i (1-p)^(n-i)
    n = len(xs)
    expected_cov = sum(
        math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(r, s)
    )
    assert result["coverage"] == float(expected_cov)
    assert result["alpha"] == float(1.0 - expected_cov)
    assert result["r"] == r
    assert result["s"] == s
    assert result["n"] == n
    assert result["p"] == p


def test_gb521_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 0.5
    r = 25
    s = 75
    result = gibbons_quantile_ci(x, p, r, s)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "upper" in result
