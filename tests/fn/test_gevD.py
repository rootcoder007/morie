"""Tests for gevD.gev_distribution."""

from morie.fn import _array_core as np

from morie.fn.gevD import gev_distribution


def test_gevD_basic():
    """Test basic functionality of gev_distribution using Coles 2001 eq. 3.2.

    GEV CDF: F(x) = exp(-(1 + xi*((x-mu)/sigma))^(-1/xi)) for xi != 0
    Standard GEV density (sec. 3.3.2):
      f(x) = (1/sigma) * (1 + xi*z)^(-1/xi - 1) * exp(-(1+xi*z)^(-1/xi))
    where z = (x - mu)/sigma and (1 + xi*z) > 0.
    """
    import math

    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    mu = 0.0
    sigma = 1.0
    xi = 0.1  # scalar shape parameter (required by implementation)

    result = gev_distribution(x, mu, sigma, xi)

    # Result must contain a .payload mapping with the documented keys.
    assert hasattr(result, "payload")
    payload = result.payload

    assert "estimate" in payload  # CDF
    assert "F" in payload
    assert "pdf" in payload
    assert "logpdf" in payload
    assert "method" in payload

    # CDF: independent computation via the Coles formula.
    z = (np.array(x) - mu) / sigma
    t = 1.0 + xi * z
    # Only evaluate where t > 0 (GEV support constraint).
    expected_F = [math.exp(-(ti ** (-1.0 / xi))) if ti > 0 else float("nan")
                  for ti in t]
    expected_logpdf = [
        (-math.log(sigma)
         - (1.0 / xi + 1.0) * math.log(ti)
         - ti ** (-1.0 / xi))
        if ti > 0 else float("-inf")
        for ti in t
    ]

    got_F = list(payload["F"])
    got_logpdf = list(payload["logpdf"])

    assert len(got_F) == len(expected_F)
    for g, e in zip(got_F, expected_F):
        assert abs(g - e) < 1e-10

    for g, e in zip(got_logpdf, expected_logpdf):
        assert abs(g - e) < 1e-10

    # logpdf vs pdf consistency via exp, guarded against underflow.
    got_pdf = list(payload["pdf"])
    for lp, p in zip(got_logpdf, got_pdf):
        if lp > -700:
            assert abs(p - math.exp(lp)) < 1e-12
        else:
            assert p == 0.0


def test_gevD_edge():
    """Test edge cases: scalar x and values at the distribution's mode."""
    mu = 0.0
    sigma = 1.0
    xi = 0.0  # Gumbel limit case; formula still finite.

    # Scalar x: function should return a RichResult with a payload.
    result = gev_distribution(1.0, mu, sigma, xi)
    assert hasattr(result, "payload")
    assert "F" in result.payload
    assert "pdf" in result.payload
    assert "logpdf" in result.payload

    # As |x - mu| -> inf inside the support, CDF -> 1 (xi > 0) or -> 1 (xi = 0).
    # For xi < 0 the distribution has an upper bound at mu - sigma/xi; we only
    # check xi >= 0 here to avoid violating the support constraint.
    x = 5.0
    result2 = gev_distribution(x, mu, sigma, xi)
    F_x = result2.payload["F"]
    # Standard GEV/Gumbel CDF at x far above the location approaches 1.
    assert 0.0 < F_x < 1.0
