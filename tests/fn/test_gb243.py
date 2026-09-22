"""Tests for gb243.gibbons_order_beta."""

from morie.fn.gb243 import gibbons_order_beta


def test_gb243_basic():
    """Test basic functionality."""
    r = 10
    n = 100
    u = 0.25
    result = gibbons_order_beta(u, r, n)
    assert isinstance(result, dict)
    # Documented keys
    for key in ("pdf", "cdf", "mean", "var", "alpha", "beta", "r", "n", "method"):
        assert key in result

    # Mean formula: E[U_(r)] = r / (n + 1)
    expected_mean = r / (n + 1)
    assert abs(result["mean"] - expected_mean) < 1e-12

    # Alpha and beta parameters of the underlying Beta(r, n - r + 1)
    assert result["alpha"] == float(r)
    assert result["beta"] == float(n - r + 1)
    assert result["r"] == r
    assert result["n"] == n

    # PDF via the theorem's explicit formula:
    # f(u) = n! / ((r-1)! (n-r)!) * u^(r-1) * (1-u)^(n-r)
    from math import factorial
    coef = factorial(n) / (factorial(r - 1) * factorial(n - r))
    expected_pdf = coef * (u ** (r - 1)) * ((1.0 - u) ** (n - r))
    assert abs(result["pdf"] - expected_pdf) < 1e-12

    # CDF: P(U_(r) <= u) = Beta(r, n-r+1).cdf(u)
    from scipy.stats import beta
    expected_cdf = float(beta.cdf(u, float(r), float(n - r + 1)))
    assert abs(result["cdf"] - expected_cdf) < 1e-12


def test_gb243_edge():
    """Test edge cases."""
    r = 10
    n = 100
    u = 0.0
    result = gibbons_order_beta(u, r, n)
    assert isinstance(result, dict)
    assert "pdf" in result
    assert "mean" in result
