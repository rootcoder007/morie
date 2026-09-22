"""Tests for gb5414.gibbons_sign_power."""

from math import comb, sqrt

from morie.fn import _array_core as np

from morie.fn.gb5414 import gibbons_sign_power


def _normal_cdf(x):
    """Standard normal CDF via the error function."""
    from math import erf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_ppf(p):
    """Inverse standard normal CDF (Beasley-Springer-Moro approximation)."""
    from math import log, sqrt
    # Rational approximation for the lower tail; symmetric via 1 - F(-z).
    a = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ]
    pl = 0.02425
    pu = 1.0 - pl
    if p < pl:
        q = sqrt(-2.0 * log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= pu:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = sqrt(-2.0 * log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def _expected_power(n, theta, alpha):
    """Independent computation of the literature formula."""
    za = _normal_ppf(1.0 - alpha)
    approx = 1.0 - _normal_cdf(
        (n * (0.5 - theta) + 0.5 * sqrt(n) * za)
        / sqrt(n * theta * (1.0 - theta))
    )
    half = 0.5 ** n
    ka = n
    for c in range(n + 1):
        tail = sum(comb(n, i) for i in range(c, n + 1)) * half
        if tail <= alpha:
            ka = c
            break
    pex = sum(
        comb(n, i) * theta ** i * (1.0 - theta) ** (n - i)
        for i in range(ka, n + 1)
    )
    return approx, ka, pex


def test_gb5414_basic():
    """Test basic functionality with the documented signature."""
    n = 100
    theta = 0.7
    alpha = 0.05
    result = gibbons_sign_power(n, theta, alpha)

    assert isinstance(result, dict)
    assert "power" in result
    assert "power_exact" in result
    assert "k_alpha" in result
    assert "alpha_exact" in result
    assert "n" in result
    assert "theta" in result
    assert "method" in result

    expected_approx, expected_ka, expected_pex = _expected_power(
        n, theta, alpha
    )
    assert abs(result["power"] - expected_approx) < 1e-10
    assert abs(result["power_exact"] - expected_pex) < 1e-12
    assert result["k_alpha"] == expected_ka
    assert result["n"] == n
    assert result["theta"] == theta


def test_gb5414_edge():
    """Test edge cases with valid inputs."""
    n = 10
    theta = 0.3
    alpha = 0.05
    result = gibbons_sign_power(n, theta, alpha, exact=False)

    assert isinstance(result, dict)
    assert "power" in result
    assert "power_exact" in result
    assert result["n"] == n
    assert result["theta"] == theta

    za = _normal_ppf(1.0 - alpha)
    expected_approx = 1.0 - _normal_cdf(
        (n * (0.5 - theta) + 0.5 * sqrt(n) * za)
        / sqrt(n * theta * (1.0 - theta))
    )
    assert abs(result["power"] - expected_approx) < 1e-10
