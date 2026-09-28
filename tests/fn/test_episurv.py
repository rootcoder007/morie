"""episurv: detectors, regressions and structures recomputed from their definitions."""

import math

import pytest

from morie.fn._qpcore import inverse
from morie.fn._rng import random_normal, random_uniform
from morie.fn.episurv import (
    bayes_outbreak,
    buffer_exposure,
    bym2_structure,
    cusum_surveillance,
    ecological_regression,
    kernel_exposure,
    leroux_precision,
)

U = [float(v) for v in random_uniform(300, seed=31)]


def _nb_cdf(y, size, prob):
    return sum(
        math.exp(
            math.lgamma(j + size)
            - math.lgamma(size)
            - math.lgamma(j + 1)
            + size * math.log(prob)
            + j * math.log(1 - prob)
        )
        for j in range(y + 1)
    )


def test_bayes_outbreak_quantile():
    obs = [int(5 + 5 * v) for v in U[:30]]
    r = bayes_outbreak(obs, w=5)
    for t, ub in zip(r.time_points, r.upperbound):
        base = obs[t - 6 : t - 1]
        size, prob = sum(base) + 0.5, 5 / 6
        assert _nb_cdf(ub, size, prob) >= 0.95 * (1 - 1e-13)
        assert ub == 0 or _nb_cdf(ub - 1, size, prob) < 0.95
    assert r.alarm == [obs[t - 1] > u for t, u in zip(r.time_points, r.upperbound)]


def test_cusum_recursion_and_regions():
    obs = [int(3 + 4 * v) for v in U[30:70]]
    r = cusum_surveillance(obs, start=20, k=0.5, h=3.0)
    m = sum(obs[:19]) / 19
    s = 0.0
    for i, t in enumerate(range(19, 40)):
        s = max(0.0, s + (obs[t] - m) / math.sqrt(m) - 0.5)
        assert r.cusum[i] == pytest.approx(s, rel=1e-13, abs=1e-15)
        assert r.alarm[i] == (s >= 3.0)
    ex = [[2.0 + c, 5.5 - 0.5 * c] for c in range(6)]
    ob = [[3, 4], [4, 5], [2, 7], [8, 3], [9, 2], [9, 1]]
    rr = cusum_surveillance(ob, ex, k=0.2, h=2.0)
    one = cusum_surveillance([row[0] for row in ob], [row[0] for row in ex], k=0.2, h=2.0)
    assert [row[0] for row in rr.cusum] == one.cusum


def test_ecological_regressions_score_equations():
    n = 30
    z = [float(v) for v in random_normal(n, seed=32)]
    e = [2.0 + 4 * U[100 + i] for i in range(n)]
    y = [int(e[i] * math.exp(0.1 + 0.5 * z[i]) * (0.1 + 2.0 * U[150 + i]) ** 2) for i in range(n)]
    X = [[v] for v in z]
    p = ecological_regression(y, e, X)
    mu = p.fitted
    assert abs(sum(a - b for a, b in zip(y, mu))) < 1e-8
    assert abs(sum((a - b) * v for a, b, v in zip(y, mu, z))) < 1e-8
    nb = ecological_regression(y, e, X, "negbin")
    w = [(a - b) / (1 + b / nb.theta) for a, b in zip(y, nb.fitted)]
    assert abs(sum(w)) < 1e-6 and abs(sum(v * c for v, c in zip(w, z))) < 1e-6
    y0 = [0 if U[200 + i] < 0.25 else v for i, v in enumerate(y)]
    zp = ecological_regression(y0, e, X, "zip")
    pz = ecological_regression(y0, e, X, "poisson")
    assert zp.loglik >= pz.loglik - 1e-9


def test_area_structures():
    A = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
    Q = leroux_precision(A, 1.0)
    assert [sum(r) for r in Q] == pytest.approx([0.0] * 4, abs=1e-15)
    assert leroux_precision(A, 0.0, 3.0) == [[3.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    b = bym2_structure(A, tau=2.0, phi=0.4)
    G = b.generalized_inverse
    Qi = [[(sum(A[i]) if i == j else 0) - A[i][j] for j in range(4)] for i in range(4)]
    QGQ = [[sum(Qi[i][k] * G[k][m] * Qi[m][j] for k in range(4) for m in range(4)) for j in range(4)] for i in range(4)]
    for i in range(4):
        for j in range(4):
            assert QGQ[i][j] == pytest.approx(Qi[i][j], abs=1e-12)
    assert math.exp(sum(math.log(G[i][i] / b.scaling_factor) for i in range(4)) / 4) == pytest.approx(1.0, rel=1e-12)
    assert b.covariance[0][0] == pytest.approx((0.6 + 0.4 * G[0][0] / b.scaling_factor) / 2.0, rel=1e-13)
    assert len(inverse(b.covariance)) == 4


def test_exposures():
    R = [[0.2, 0.3], [0.7, 0.9]]
    S = [[0.25, 0.35], [0.6, 0.8], [0.1, 0.95]]
    r = buffer_exposure(R, S, 0.2, [1.0, 2.0, 5.0])
    assert r.count == [1, 1] and r.weighted == [1.0, 2.0]
    q = kernel_exposure(R, S, 0.5, kernel="quartic")
    d = [math.dist(R[0], s) / 0.5 for s in S]
    assert q[0] == pytest.approx(sum(3 / math.pi * (1 - v * v) ** 2 for v in d if v < 1) / 0.25, rel=1e-13)
