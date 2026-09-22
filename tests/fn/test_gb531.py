"""Tests for gb531.gibbons_quantile_test."""

from morie.fn import _array_core as np

from morie.fn.gb531 import gibbons_quantile_test


def test_gb531_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100).tolist()
    q0 = 0.0
    p = 0.5
    alternative = "two-sided"
    result = gibbons_quantile_test(x, q0, p, alternative)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "n" in result
    assert "p" in result
    assert "mean" in result
    assert "var" in result
    assert "alternative" in result
    assert "method" in result

    n = len(x)
    assert result["n"] == n
    assert result["p"] == p
    assert result["mean"] == n * p
    assert result["var"] == n * p * (1.0 - p)
    assert result["alternative"] == alternative

    k = sum(1 for v in x if v <= q0)
    assert result["statistic"] == k

    # Independent computation of the p-value using the documented formula.
    from math import comb

    def pmf(i):
        return comb(n, i) * p ** i * (1.0 - p) ** (n - i)

    lower = sum(pmf(i) for i in range(k + 1))
    upper = sum(pmf(i) for i in range(k, n + 1))
    expected_pv = min(1.0, 2.0 * min(lower, upper))
    assert abs(result["p_value"] - expected_pv) < 1e-12


def test_gb531_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100).tolist()
    q0 = 0.0
    p = 0.5
    alternative = "two-sided"
    result = gibbons_quantile_test(x, q0, p, alternative)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert result["n"] == len(x)
    assert 0.0 <= result["p_value"] <= 1.0
