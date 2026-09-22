"""Tests for gb5712.gibbons_wsrt_power."""

from morie.fn.gb5712 import gibbons_wsrt_power


def test_gb5712_basic():
    """Test basic functionality."""
    n = 100
    p1 = 0.6
    p2 = 0.7
    alpha = 0.05
    result = gibbons_wsrt_power(n, p1, p2, alpha)
    assert isinstance(result, dict)
    assert "power" in result
    assert "z_beta" in result
    assert "shift" in result
    assert "sd0" in result
    # Independent computation of the formula
    import math
    N = n
    shift = N * (p1 - 0.5) + N * (N - 1.0) * (p2 - 0.5) / 2.0
    sd0 = math.sqrt(N * (N + 1.0) * (2.0 * N + 1.0) / 24.0)
    # z_alpha for one-sided upper-tail test with alpha=0.05 is 1.6448536269514722
    z_alpha = 1.6448536269514722
    z_beta_expected = shift / sd0 - z_alpha
    # Phi(z_beta) computed by independent expression using erf
    from math import erf, sqrt
    power_expected = 0.5 * (1.0 + erf(z_beta_expected / sqrt(2.0)))
    assert abs(result["power"] - power_expected) < 1e-9
    assert abs(result["z_beta"] - z_beta_expected) < 1e-9
    assert abs(result["shift"] - shift) < 1e-9
    assert abs(result["sd0"] - sd0) < 1e-9
    assert result["n"] == N
    assert result["p1"] == p1
    assert result["p2"] == p2


def test_gb5712_edge():
    """Test edge cases."""
    n = 100
    p1 = 0.5
    p2 = 0.5
    alpha = 0.05
    result = gibbons_wsrt_power(n, p1, p2, alpha)
    assert isinstance(result, dict)
    # Under H1 null (p1=p2=0.5), shift=0 and power = 1 - alpha for upper-tailed test
    # z_beta = 0 - z_alpha = -z_alpha, so power = Phi(-z_alpha) = alpha
    # Actually, since shift=0, z_beta = -z_alpha, power = Phi(-z_alpha) = alpha
    import math
    from math import erf, sqrt
    N = n
    shift = N * (p1 - 0.5) + N * (N - 1.0) * (p2 - 0.5) / 2.0
    sd0 = math.sqrt(N * (N + 1.0) * (2.0 * N + 1.0) / 24.0)
    z_alpha = 1.6448536269514722
    z_beta_expected = shift / sd0 - z_alpha
    power_expected = 0.5 * (1.0 + erf(z_beta_expected / sqrt(2.0)))
    assert abs(result["power"] - power_expected) < 1e-9
    # When p1=p2=0.5, shift=0, so power should be Phi(-z_alpha) = alpha
    assert abs(result["power"] - alpha) < 1e-9
    assert "method" in result
