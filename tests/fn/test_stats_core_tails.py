"""Tail accuracy of the _stats_core distributions.

Reference values marked "mpmath" were computed with mpmath at 30-50
significant digits (quadrature split at the integrand's scale); where
scipy 1.x disagrees with them (nct cdf at -40, skewnorm quantiles below
1e-15, halfnorm and beta quantiles below 1e-100, f.isf(1e-15)) the value
here is the mpmath one.
"""

import math

import pytest

from morie.fn import _stats_core as sc


def rel(a, b):
    return abs(a - b) / abs(b)


# ---- logcdf / logsf keep the digits of the small complement ------------------


def test_logcdf_near_one_is_log1p_of_the_sf():
    for dist, args, x in (
        (sc.t, (30,), 40.0),
        (sc.chi2, (1,), 100.0),
        (sc.gamma, (4.0,), 100.0),
        (sc.f, (50, 200), 20.0),
        (sc.expon, (), 100.0),
    ):
        sf = dist.sf(x, *args)
        assert 0.0 < sf < 1e-10
        assert rel(dist.logcdf(x, *args), math.log1p(-sf)) <= 1e-12


def test_logsf_near_zero_is_log1p_of_the_cdf():
    for dist, args, x in ((sc.chi2, (300,), 20.0), (sc.gamma, (4.0,), 1e-10), (sc.beta, (30, 0.7), 0.05)):
        c = dist.cdf(x, *args)
        assert 0.0 < c < 1e-10
        assert rel(dist.logsf(x, *args), math.log1p(-c)) <= 1e-12


# ---- closed-form tails --------------------------------------------------------


def test_closed_form_tails():
    assert rel(sc.laplace.sf(40.0), 0.5 * math.exp(-40.0)) <= 1e-15
    assert rel(sc.logistic.sf(40.0), 1.0 / (1.0 + math.exp(40.0))) <= 1e-15
    assert rel(sc.cauchy.cdf(-1e10), math.atan(1e-10) / math.pi) <= 1e-15
    assert rel(sc.weibull_min.cdf(1e-10, 1.7), -math.expm1(-((1e-10) ** 1.7))) <= 1e-15
    assert rel(sc.weibull_min.sf(20.0, 1.7), math.exp(-(20.0**1.7))) <= 1e-15
    assert rel(sc.rayleigh.sf(30.0), math.exp(-450.0)) <= 1e-15
    assert rel(sc.lognorm.sf(1e4, 0.7), sc.norm.sf(math.log(1e4) / 0.7)) <= 1e-15
    assert rel(sc.halfnorm.logsf(1e-10), math.log1p(-math.erf(1e-10 / math.sqrt(2.0)))) <= 1e-15
    assert rel(sc.genpareto.sf(1e6, 0.3), (1 + 0.3e6) ** (-1 / 0.3)) <= 1e-13
    assert rel(sc.binom.sf(20, 1000, 0.001), sc._betainc(21.0, 980.0, 0.001)) <= 1e-15
    assert sc.betabinom.sf(10, 10, 2.0, 3.0) == 0.0
    assert sc.hypergeom.sf(10, 50, 10, 12) == 0.0


def test_mpmath_references():
    # mpmath, split quadrature of 2 phi(t) Phi(3t) over (-inf, -12]
    assert rel(sc.skewnorm.logcdf(-12.0, 3.0), -729.5178956330957) <= 1e-12
    # mpmath, conditioning on V ~ chi2(30)
    assert rel(sc.nct.cdf(-12.0, 30, 3.0), 4.460469536643723e-21) <= 1e-12
    # mpmath, Poisson mixture of regularized upper gammas
    assert rel(sc.ncx2.sf(100.0, 4, 2.5), 3.002445188299769e-16) <= 1e-12
    # mpmath, E[theta^2] of the von Mises density on (-pi, pi]
    assert rel(sc.vonmises.var(2.0), 0.7644618798111269) <= 1e-13
    assert rel(sc.vonmises.var(20.0), 0.051323846749615582) <= 1e-13


# ---- quantiles solve the tail they are asked for ------------------------------


@pytest.mark.parametrize("q", [1e-300, 1e-100, 1e-15])
def test_extreme_quantiles_invert_their_own_tail(q):
    for dist, args in ((sc.f, (3, 7)), (sc.nct, (5, 1.5)), (sc.ncx2, (4, 2.5)), (sc.skewnorm, (3.0,))):
        x = dist.isf(q, *args)
        assert rel(dist.sf(x, *args), q) <= 1e-9
    for dist, args in ((sc.beta, (2.0, 5.0)), (sc.skewnorm, (3.0,)), (sc.nct, (5, 1.5))):
        x = dist.ppf(q, *args)
        assert rel(dist.cdf(x, *args), q) <= 1e-9


def test_halfnorm_small_quantile_is_the_erf_inverse():
    # erf(z / sqrt 2) = q; for tiny q, z = q sqrt(pi/2) to all digits
    assert rel(sc.halfnorm.ppf(1e-100), 1e-100 * math.sqrt(math.pi / 2.0)) <= 1e-15
    z = sc.halfnorm.ppf(1e-3)
    assert rel(math.erf(z / math.sqrt(2.0)), 1e-3) <= 1e-14


def test_nct_power_tail():
    # P(T <= x) ~ C |x|^-df: 1e-300 sits near -6.8e59 for df = 5
    x = sc.nct.ppf(1e-300, 5, 1.5)
    assert -1e61 < x < -1e59
    assert rel(sc.nct.cdf(x, 5, 1.5), 1e-300) <= 1e-9


def test_isf_above_one_half_uses_the_exact_complement():
    q = 1.0 - 1e-9
    x = sc.ncx2.isf(q, 4, 2.5)
    assert rel(sc.ncx2.cdf(x, 4, 2.5), 1.0 - q) <= 1e-9


# ---- closed-form moments -----------------------------------------------------


def test_closed_form_moments():
    assert sc.t.mean(3.5) == 0.0 and math.isnan(sc.t.mean(1))
    assert sc.t.var(30) == 30 / 28 and sc.t.var(1.5) == math.inf and math.isnan(sc.t.var(1))
    assert sc.f.mean(3, 7) == 7 / 5
    assert rel(sc.f.var(3, 7), 2 * 49 * 8 / (3 * 25 * 3)) <= 1e-15
    assert sc.chi2.mean(7.5) == 7.5 and sc.chi2.var(7.5) == 15.0
    assert sc.gamma.var(0.3) == 0.3
    assert sc.expon.var(0, 2.5) == 6.25
    assert rel(sc.weibull_min.mean(1.7), math.gamma(1 + 1 / 1.7)) <= 1e-15
    assert rel(sc.lognorm.var(0.7), math.expm1(0.49) * math.exp(0.49)) <= 1e-15
    assert rel(sc.gumbel_r.var(), math.pi**2 / 6) <= 1e-15
    assert sc.laplace.mean() == 0.0 and sc.logistic.mean() == 0.0 and sc.vonmises.mean(2.0) == 0.0
    d = 3.0 / math.sqrt(10.0)
    assert rel(sc.skewnorm.mean(3.0), d * math.sqrt(2 / math.pi)) <= 1e-15
    m = 1.5 * math.sqrt(2.5) * math.exp(math.lgamma(2.0) - math.lgamma(2.5))
    assert rel(sc.nct.mean(5, 1.5), m) <= 1e-15
    assert rel(sc.nct.var(5, 1.5), 5 * (1 + 2.25) / 3 - m * m) <= 1e-15


def test_anderson_ksamp_pvalue_is_the_quadratic_fit():
    a = [math.sin(1.3 * i) for i in range(40)]
    b = [0.25 + math.cos(0.7 * i) for i in range(35)]
    r = sc.anderson_ksamp((a, b))
    tm = r.critical_values
    ls = [math.log(s) for s in (0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001)]
    S = [[sum(t ** (i + j) for t in tm) for j in range(3)] for i in range(3)]
    y = [sum(lv * t**i for t, lv in zip(tm, ls)) for i in range(3)]
    c = sc._solve3(S, y)
    assert tm[0] <= r.statistic <= tm[-1]  # inside the fitted range
    assert rel(r.pvalue, math.exp(c[0] + c[1] * r.statistic + c[2] * r.statistic**2)) <= 1e-12


def test_second_batch_families():
    # closed forms (scipy loses these digits through 1 - cdf)
    assert rel(sc.fisk.sf(1e4, 3.0), 1.0 / (1.0 + 1e12)) <= 1e-15
    assert rel(sc.burr.sf(1e4, 3.0, 2.0), -math.expm1(-2.0 * math.log1p(1e-12))) <= 1e-15
    assert rel(sc.burr12.sf(1e2, 2.0, 3.0), (1.0 + 1e4) ** -3.0) <= 1e-14
    assert rel(sc.invweibull.sf(1e4, 2.5), -math.expm1(-((1e4) ** -2.5))) <= 1e-15
    assert rel(sc.nakagami.sf(5.0, 1.5), sc._gammainc_q(1.5, 37.5)) <= 1e-15
    assert rel(sc.hypsecant.sf(40.0), 2.0 / math.pi * math.atan(math.exp(-40.0))) <= 1e-15
    assert rel(sc.laplace_asymmetric.sf(40.0, 1.7), math.exp(-68.0) / (1.0 + 1.7**2)) <= 1e-14
    # log F = -log(1 + x^-c) = c log x - log1p(x^c) at x = 1e-300
    assert rel(sc.fisk.logcdf(1e-300, 3.0), 3.0 * math.log(1e-300)) <= 1e-15
    # mpmath: explicit 400-term Poisson mixture of regularized gammas
    assert rel(sc.rice.sf(20.0, 1.2), 1.5517214477224549e-78) <= 1e-12
    # mpmath: -(p^lam - (1-p)^lam)/lam at p = 1e-15, lam = 0.14
    assert rel(sc.tukeylambda.isf(1e-15, 0.14), 7.0861194118054075) <= 1e-13
    # mpmath quadrature of the Johnson SB variance
    assert rel(sc.johnsonsb.var(0.5, 1.2), 0.031303835321102109) <= 1e-10
    lam = 0.14
    v = 2 / lam**2 * (1 / (1 + 2 * lam) - math.exp(2 * math.lgamma(1 + lam) - math.lgamma(2 + 2 * lam)))
    assert rel(sc.tukeylambda.var(lam), v) <= 1e-15
