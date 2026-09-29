"""Tests for morie.fn.pwr_av: noncentral-F power recomputed by Poisson mixing of central F tails."""

import math

from morie.fn.pwr_av import power_anova


def _beta_inc(a, b, x, m=20000):
    # regularised incomplete beta by the midpoint rule on t = x u (small smooth integrand)
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    h = x / m
    return sum(math.exp((a - 1) * math.log((i + 0.5) * h) + (b - 1) * math.log(1 - (i + 0.5) * h) - lb) for i in range(m)) * h


def _ncf_sf(crit, d1, d2, ncp):
    # P(F' > c) = sum_j Pois(j; ncp/2) * (1 - I_x(d1/2 + j, d2/2)), x = d1 c / (d1 c + d2)
    x = d1 * crit / (d1 * crit + d2)
    s = 0.0
    for j in range(60):
        w = math.exp(-ncp / 2) * (ncp / 2) ** j / math.factorial(j)
        s += w * (1 - _beta_inc(d1 / 2 + j, d2 / 2, x))
    return s


def test_power_matches_noncentral_f():
    from morie.fn._stats_core import f as fdist

    n, k, f = 10, 3, 0.4
    d1, d2 = k - 1, k * (n - 1)
    crit = float(fdist.ppf(0.95, d1, d2))
    assert abs(_ncf_sf(crit, d1, d2, 0.0) - 0.05) < 1e-6
    assert abs(power_anova(n=n, k=k, f=f) - _ncf_sf(crit, d1, d2, f * f * n * k)) < 1e-6


def test_solutions_reproduce_power():
    n = power_anova(k=4, f=0.25, power=0.8)
    assert abs(power_anova(n=n, k=4, f=0.25) - 0.8) < 1e-10
    f = power_anova(n=12, k=3, power=0.7)
    assert abs(power_anova(n=12, k=3, f=f) - 0.7) < 1e-10
    k = power_anova(n=30, f=0.3, power=0.9)
    assert power_anova(n=30, k=int(k), f=0.3) >= 0.9 > power_anova(n=30, k=int(k) - 1, f=0.3) if k > 2 else True
