"""Tests for betbnm.beta_binomial."""

from morie.fn import _array_core as np

from morie.fn.betbnm import beta_binomial


def test_betbnm_basic():
    """Test basic functionality against the documented Beta-Binomial formulas."""
    y, n = 7, 12
    alpha, beta = 0.05, 0.8

    result = beta_binomial(y, n, alpha, beta)

    # The function returns a RichResult (dict-like) with these documented keys.
    expected_keys = {
        "postalpha", "postbeta", "postmean", "postvar", "postmode",
        "priormean", "logmarglik", "predmean", "predvar", "m",
    }
    assert expected_keys.issubset(set(result.keys()))

    # Posterior parameters are the prior plus observed successes/failures.
    assert result["postalpha"] == alpha + y
    assert result["postbeta"] == beta + n - y

    # Posterior mean and variance from the Beta(alpha+y, beta+n-y) formulas.
    pa, pb = alpha + y, beta + n - y
    s = pa + pb
    postmean = pa / s
    postvar = pa * pb / (s * s * (s + 1.0))
    assert result["postmean"] == postmean
    assert result["postvar"] == postvar

    # Prior mean from the Beta(alpha, beta) formula.
    assert result["priormean"] == alpha / (alpha + beta)

    # m defaults to n when not provided.
    assert result["m"] == n

    # Predictive quantities (with m == n) are functions of the posterior mean.
    assert result["predmean"] == n * postmean
    predvar_expected = n * postmean * (1.0 - postmean) * (n + s) / (s + 1.0)
    assert result["predvar"] == predvar_expected

    # log marginal likelihood: lgamma(n+1) - lgamma(y+1) - lgamma(n-y+1)
    # + lbeta(alpha+y, beta+n-y) - lbeta(alpha, beta).
    from math import lgamma
    def lbeta(x, z):
        return lgamma(x) + lgamma(z) - lgamma(x + z)
    log_marg_expected = (
        lgamma(n + 1.0) - lgamma(y + 1.0) - lgamma(n - y + 1.0)
        + lbeta(pa, pb) - lbeta(alpha, beta)
    )
    assert result["logmarglik"] == log_marg_expected


def test_betbnm_edge():
    """Test edge cases: alpha=1, beta=1 uniform prior, no observations, m explicit."""
    y, n = 0, 0
    alpha, beta = 1.0, 1.0
    m = 5

    result = beta_binomial(y, n, alpha, beta, m=m)

    # With no data the posterior is just the prior.
    assert result["postalpha"] == alpha
    assert result["postbeta"] == beta
    assert result["postmean"] == 0.5
    assert result["priormean"] == 0.5

    # Postmode is NaN when either posterior shape is <= 1.
    import math
    assert math.isnan(result["postmode"])

    # Predictive uses the supplied m rather than defaulting to n.
    assert result["m"] == m
    assert result["predmean"] == m * result["postmean"]
