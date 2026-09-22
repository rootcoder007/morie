"""Tests for gpdD.gpd_distribution."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gpdD import gpd_distribution


def test_gpdD_basic():
    """Test basic functionality."""
    sigma = 1.0
    xi = 0.5
    result = gpd_distribution(sigma, xi)
    assert isinstance(result, dict)
    # Core keys documented by the function's payload
    for key in ("cdf", "pdf", "quantile", "mean", "variance",
                "upper_endpoint", "sigma", "xi", "n", "method"):
        assert key in result
    # sigma and xi echoed back
    assert result["sigma"] == sigma
    assert result["xi"] == xi
    # n reflects the default x grid length
    assert result["n"] == 4
    # For xi = 0.5 < 1, mean = sigma / (1 - xi) is finite
    expected_mean = sigma / (1.0 - xi)
    assert result["mean"] == expected_mean
    # For xi = 0.5 < 0.5 is false, so variance should be inf
    assert result["variance"] == float("inf")
    # With xi = 0.5 >= 0, the upper endpoint is +inf
    assert result["upper_endpoint"] == float("inf")
    # Quantiles are computed for the default p grid (length 4)
    assert len(result["quantile"]) == 4
    # CDF and PDF are computed for the default x grid (length 4)
    assert len(result["cdf"]) == 4
    assert len(result["pdf"]) == 4


def test_gpdD_edge():
    """Test edge cases."""
    sigma = 1.0
    xi = 0.5
    # Explicit x and p grids of the same length; both must be list-like of numbers
    x_vals = [0.5, 1.0, 2.0, 4.0]
    p_vals = [0.5, 0.9, 0.95, 0.99]
    result = gpd_distribution(sigma, xi, x=x_vals, p=p_vals)
    assert isinstance(result, dict)
    assert result["n"] == len(x_vals)
    assert len(result["cdf"]) == len(x_vals)
    assert len(result["pdf"]) == len(x_vals)
    assert len(result["quantile"]) == len(p_vals)
    # For xi = 0.5 >= 0, the upper endpoint is +inf
    assert result["upper_endpoint"] == float("inf")
    # CDF values must lie in [0, 1] (Pickands GPD: F(x) = 1 - (1 + xi x/sigma)^(-1/xi))
    for v in result["cdf"]:
        assert 0.0 <= v <= 1.0
    # Quantiles must be non-negative for the standard GPD support
    for q in result["quantile"]:
        assert q >= 0.0
    # Independent CDF check at x = 1: F(1) = 1 - (1 + 0.5*1/1)^(-1/0.5)
    xi_f = float(xi)
    s_f = float(sigma)
    expected_cdf_at_1 = 1.0 - (1.0 + xi_f * 1.0 / s_f) ** (-1.0 / xi_f)
    assert result["cdf"][x_vals.index(1.0)] == expected_cdf_at_1
