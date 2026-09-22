"""Tests for gb_mnq.gibbons_marginal_quant."""

from morie.fn import _array_core as np

from morie.fn.gb_mnq import gibbons_marginal_quant


def test_gb_mnq_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    p = 0.25
    result = gibbons_marginal_quant(x, p)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result
    # Documented keys should all be present.
    for key in ("estimate", "r", "n", "p", "u_mean", "u_var", "method"):
        assert key in result
    # n should match input length; p should be preserved (float).
    assert result["n"] == 100
    assert result["p"] == p
    # r = floor(n*p) + 1.
    r_expected = int(np.floor(100 * p)) + 1
    assert result["r"] == r_expected
    # u_mean = r / (n+1) and u_var = r*(n-r+1)/((n+1)^2*(n+2)).
    n = 100
    u_mean_expected = r_expected / (n + 1.0)
    u_var_expected = (
        r_expected * (n - r_expected + 1.0)
        / ((n + 1.0) ** 2 * (n + 2.0))
    )
    assert result["u_mean"] == u_mean_expected
    assert result["u_var"] == u_var_expected
    # estimate is the (r-1)-th order statistic (0-indexed).
    sorted_x = sorted(float(v) for v in x)
    assert result["estimate"] == sorted_x[r_expected - 1]


def test_gb_mnq_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    p = 0.75
    result = gibbons_marginal_quant(x, p)
    assert isinstance(result, dict)
    # Documented keys should all be present.
    for key in ("estimate", "r", "n", "p", "u_mean", "u_var", "method"):
        assert key in result
    assert result["n"] == 100
    assert result["p"] == p
    r_expected = int(np.floor(100 * p)) + 1
    assert result["r"] == r_expected
