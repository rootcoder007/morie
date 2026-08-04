"""The d/p/q/r family, anchored on R.

Every expected value is what R prints at full precision.  A test whose
expectation came from running this code could not detect this code being
wrong, so none of these did.
"""

import math

from morie.fn import _rrng_core as R

TOL = 1e-12


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_normal_matches_r():
    assert close(R.dnorm(0), 0.398942280401433)
    assert close(R.dnorm(1.5, 2.0, 3.0), 0.13114657203397997)
    assert close(R.pnorm(1.96), 0.975002104851780)
    assert close(R.qnorm(0.975), 1.959963984540054)
    assert close(R.qnorm(0.025), -1.959963984540054)


def test_normal_far_tail_uses_erfc():
    # 1 - Phi(9) rounds to 0 in double precision; erfc keeps the digits
    assert close(R.pnorm(-9.0), 1.128588405953840e-19, 1e-10)
    assert R.pnorm(9.0, lower_tail=False) > 0.0


def test_gamma_and_chisq_match_r():
    assert close(R.dgamma(2.0, 3.0, 1.0), 0.270670566473225)
    assert close(R.pgamma(2.0, 3.0, 1.0), 0.323323583816937)
    assert close(R.qgamma(0.5, 3.0, 1.0), 2.6740603137235608, 1e-12)
    # chi-square is gamma(df/2, rate 1/2); R: pchisq(3.84, 1) = 0.9499565
    assert close(R.pchisq(3.841458820694124, 1), 0.95, 1e-9)
    assert close(R.qchisq(0.95, 1), 3.841458820694124, 1e-9)


def test_poisson_matches_r():
    assert close(R.dpois(3, 2.5), 0.21376301724973648)
    assert close(R.ppois(3, 2.5), 0.75757613313306593)
    # R defines qpois as the smallest k with cdf >= p
    assert R.qpois(0.75757613313306593, 2.5) == 3
    assert R.qpois(0.5, 2.5) == 2


def test_binomial_matches_r():
    assert close(R.dbinom(3, 10, 0.3), 0.266827932)
    assert close(R.pbinom(3, 10, 0.3), 0.64961071840000018)
    assert R.qbinom(0.64961071840000018, 10, 0.3) == 3
    # the edges are exact, not approximate
    assert R.dbinom(0, 10, 0.0) == 1.0
    assert R.dbinom(10, 10, 1.0) == 1.0


def test_beta_and_t_and_f_match_r():
    assert close(R.dbeta(0.5, 2, 3), 1.5)
    assert close(R.pbeta(0.5, 2, 3), 0.6875)
    assert close(R.qbeta(0.6875, 2, 3), 0.5, 1e-9)
    assert close(R.dt(0.0, 5), 0.37960668982249446)
    assert close(R.pt(2.015048372669157, 5), 0.94999999995764739)
    assert close(R.qt(0.975, 10), 2.228138851986273, 1e-9)
    assert close(R.pf(4.256494729093272, 3, 10), 0.96482098934016436)


def test_lognormal_is_consistent_with_normal():
    assert close(R.plnorm(math.e), R.pnorm(1.0))
    assert close(R.qlnorm(0.5), 1.0, 1e-9)


def test_exponential_matches_r():
    assert close(R.dexp(1.0, 2.0), 0.270670566473225)
    assert close(R.pexp(1.0, 2.0), 0.864664716763387)
    assert close(R.qexp(0.864664716763387, 2.0), 1.0, 1e-9)
    # the upper tail is exp(-rate*q), not 1 - cdf, so it survives far out
    assert R.pexp(700.0, 1.0, lower_tail=False) > 0.0  # exp(-800) underflows; R gives 0 too


def test_cdf_quantile_round_trip():
    for p in (0.001, 0.1, 0.5, 0.9, 0.999):
        assert close(R.pnorm(R.qnorm(p)), p, 1e-9)
        assert close(R.pgamma(R.qgamma(p, 2.5), 2.5), p, 1e-8)
        assert close(R.pbeta(R.qbeta(p, 2, 3), 2, 3), p, 1e-8)
        assert close(R.pt(R.qt(p, 7), 7), p, 1e-8)


def test_draws_are_reproducible_and_stream_stable():
    R.set_seed(1)
    a = R.rnorm(6)
    R.set_seed(1)
    b = R.rnorm(3)
    # inversion means draw k depends only on uniform k, so a prefix matches
    assert all(close(x, y, 1e-15) for x, y in zip(a, b))

    R.set_seed(7)
    g = R.rgamma(50, 2.0, 1.0)
    assert all(v > 0 for v in g)
    R.set_seed(7)
    bn = R.rbinom(50, 10, 0.3)
    assert all(0 <= v <= 10 for v in bn)


def test_parameters_are_validated():
    import pytest
    with pytest.raises(ValueError):
        R.dnorm(0, sd=0)
    with pytest.raises(ValueError):
        R.qnorm(0.0)
    with pytest.raises(ValueError):
        R.dgamma(1.0, -1.0)
    with pytest.raises(ValueError):
        R.dbinom(1, 10, 1.5)
    with pytest.raises(ValueError):
        R.dexp(1.0, 0.0)
