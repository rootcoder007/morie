"""Tests for probkit: probability and inference toolkit."""

import math

from morie.fn._rng import random_uniform
from morie.fn.probkit import (
    bv_logistic_simulate,
    chisq1_cdf,
    crps_cdf,
    ddm_drift,
    folded_normal,
    harmonic_mean_evidence,
    king_kinship,
    max_entropy_discrete,
    product_variance,
    weibull_moments,
)


def _P(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def test_product_variance_and_weibull_moments():
    assert product_variance(1.3, 0.4, -2.2, 1.7) == 0.4 * 1.7 + 1.3**2 * 1.7 + 2.2**2 * 0.4
    r = weibull_moments(0.7, 2.3)
    m = math.gamma(1 + 1 / 2.3) / 0.7 ** (1 / 2.3)
    assert abs(r.mean - m) <= 1e-12
    assert abs(r.var - (math.gamma(1 + 2 / 2.3) / 0.7 ** (2 / 2.3) - m * m)) <= 1e-12


def test_folded_normal_and_chisq1():
    r = folded_normal([0.9], 0.7, 1.3)
    assert abs(r.cdf[0] - (_P((0.9 - 0.7) / 1.3) + _P((0.9 + 0.7) / 1.3) - 1)) <= 1e-15
    assert abs(folded_normal([2.0]).cdf[0] - (2 * _P(2.0) - 1)) <= 1e-15
    assert abs(folded_normal([1.0]).mean - math.sqrt(2 / math.pi)) <= 1e-15
    assert abs(folded_normal([1.0]).var - (1 - 2 / math.pi)) <= 1e-15
    assert abs(chisq1_cdf([3.0])[0] - math.erf(math.sqrt(1.5))) <= 1e-15


def test_harmonic_mean_evidence():
    ll = [-10.5, -12.25, -9.8]
    want = math.log(1 / (sum(math.exp(-v) for v in ll) / 3))
    assert abs(harmonic_mean_evidence(ll).log_evidence - want) <= 1e-12


def test_crps_matches_closed_forms():
    for y in (0.3, -1.7, 2.9):
        want = y * (2 * _P(y) - 1) + 2 * math.exp(-y * y / 2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
        assert abs(crps_cdf(_P, y) - want) <= 1e-12
    lam, y = 2.0, 0.8
    want = y + 2 / lam * math.exp(-lam * y) - 3 / (2 * lam)
    assert abs(crps_cdf(lambda z: 1 - math.exp(-lam * z) if z > 0 else 0.0, y, breaks=(0,)) - want) <= 1e-12
    # uniform(0, 3) at y: (y^3 + (3 - y)^3) / 27 ... = int_0^y (z/3)^2 + int_y^3 (1 - z/3)^2
    y = 2.2
    want = y**3 / 27 + (3 - y) ** 3 / 27
    assert abs(crps_cdf(lambda z: min(max(z / 3, 0.0), 1.0), y, breaks=(0, 3)) - want) <= 1e-12


def test_max_entropy_moments_and_gibbs_form():
    G = [[1, 2, 3, 4, 5, 6], [1, 4, 9, 16, 25, 36]]
    r = max_entropy_discrete(G, [4.1, 19.0])
    assert abs(sum(r.p) - 1) <= 1e-12
    for k, t in enumerate([4.1, 19.0]):
        assert abs(sum(p * g for p, g in zip(r.p, G[k])) - t) <= 1e-10
    lp = [math.log(p) for p in r.p]
    for i in range(1, 6):
        assert abs(lp[i] - lp[0] - sum(r.lambdas[k] * (G[k][i] - G[k][0]) for k in range(2))) <= 1e-10


def test_bv_logistic_reproduces_shi_draws():
    r = bv_logistic_simulate(25, 0.45, seed=9)
    u, mix, e1, e2 = (random_uniform(25, seed=9, stream=s) for s in range(4))
    for i in range(25):
        z = -math.log(float(e1[i])) + (-math.log(float(e2[i])) if float(mix[i]) < 0.45 else 0.0)
        assert abs(r.x[i] - math.log(1 / (z * float(u[i]) ** 0.45))) <= 1e-12
        assert abs(r.y[i] - math.log(1 / (z * (1 - float(u[i])) ** 0.45))) <= 1e-12


def test_bv_logistic_margins_are_gumbel():
    r = bv_logistic_simulate(4000, 0.6, seed=3)
    for v in (r.x, r.y):
        f = sum(1 for a in v if a <= 0.5) / 4000
        want = math.exp(-math.exp(-0.5))
        assert abs(f - want) <= 4 * math.sqrt(want * (1 - want) / 4000)


def test_ddm_detects_rate_jump():
    e = [1 if (i * 37) % 10 < 1 else 0 for i in range(300)] + [1 if (i * 37) % 10 < 6 else 0 for i in range(300, 600)]
    r = ddm_drift(e)
    assert r.drifts and 300 <= r.drifts[0] < 400
    assert ddm_drift([0, 1] * 200).drifts == []


def test_king_kinship_identity_and_opposite_homozygotes():
    g = [[0, 1, 2, 1, 1, 0], [0, 1, 2, 1, 1, 0], [2, 1, 0, 1, 0, 2]]
    K = king_kinship(g).kinship
    assert K[0][1] == 0.5
    # pair (0, 2): hets 3 and 2, shared hets 2, opposite homozygotes 3
    assert abs(K[0][2] - ((2 - 6) / (2 * 2) + 0.5 - 5 / 8)) <= 1e-15
    assert abs(king_kinship(g, method="within").kinship[0][2] - (2 - 6) / 5) <= 1e-15
