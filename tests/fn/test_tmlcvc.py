"""Tests for tmlcvc.cvtmle (cross-validated TMLE of the ATE)."""

import math
import statistics

import pytest


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p):
    return math.log(p / (1.0 - p))


def _data(n=40):
    """Deterministic W, A, Y and deliberately imperfect initial fits."""
    W = [math.sin(1.7 * k) for k in range(n)]
    g1 = [_expit(0.3 + 0.8 * w) for w in W]
    A = [1.0 if ((37 * k + 11) % 97 + 0.5) / 97.0 < g else 0.0 for k, g in enumerate(g1)]
    Y = [1.0 if ((53 * k + 7) % 89 + 0.5) / 89.0 < _expit(-0.5 + a + 0.9 * w) else 0.0
         for k, (a, w) in enumerate(zip(A, W))]
    Q1 = [_expit(-0.3 + 1.0 + 0.6 * w) for w in W]
    Q0 = [_expit(-0.3 + 0.6 * w) for w in W]
    QA = [q1 if a else q0 for a, q1, q0 in zip(A, Q1, Q0)]
    gh = [_expit(0.2 + 0.7 * w) for w in W]
    return Y, A, QA, Q1, Q0, gh


def _target(Y, A, QA, Q1, Q0, g1W, gb=0.025):
    """tmle 2.1.1: logistic fluctuation on H1 = A/g1 and H0 = (1-A)/g0
    with offset logit(Q(A,W)); the two supports are disjoint, so each
    epsilon solves its own monotone score equation (bisection here)."""
    g1 = [min(max(g, gb), 1 - gb) for g in g1W]
    g0 = [1 - g for g in g1]

    def solve(arm, g):
        idx = [i for i, a in enumerate(A) if a == arm]

        def score(e):
            return sum((Y[i] - _expit(_logit(QA[i]) + e / g[i])) / g[i] for i in idx)
        lo, hi = -50.0, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if score(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    e1, e0 = solve(1.0, g1), solve(0.0, g0)
    Q1s = [_expit(_logit(q) + e1 / g) for q, g in zip(Q1, g1)]
    Q0s = [_expit(_logit(q) + e0 / g) for q, g in zip(Q0, g0)]
    QAs = [q1 if a else q0 for a, q1, q0 in zip(A, Q1s, Q0s)]
    mu1, mu0 = statistics.fmean(Q1s), statistics.fmean(Q0s)
    IC1 = [a / g * (y - q) + q1 - mu1 for a, g, y, q, q1 in zip(A, g1, Y, QAs, Q1s)]
    IC0 = [(1 - a) / g * (y - q) + q0 - mu0 for a, g, y, q, q0 in zip(A, g0, Y, QAs, Q0s)]
    return mu1, mu0, IC1, IC0, (e0, e1), g1, Q1s, Q0s, QAs

from morie.fn.tmlcvc import cvtmle, tmle_cv_targeting


def _bisect(f, lo=-50.0, hi=50.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cv(Y, A, Q1, Q0, g, fold):
    """Per fold: one epsilon on H = A/g - (1-A)/(1-g) solving the logistic
    score (monotone, so bisection); psi = mean over folds of the fold
    means; IC at the targeted fit, centred at the fold estimate."""
    n = len(Y)
    psis, ic = [], [0.0] * n
    for f in sorted(set(fold)):
        idx = [i for i in range(n) if fold[i] == f]
        H = {i: A[i] / g[i] - (1 - A[i]) / (1 - g[i]) for i in idx}
        QA = {i: Q1[i] if A[i] else Q0[i] for i in idx}
        e = _bisect(lambda e: sum(H[i] * (Y[i] - _expit(_logit(QA[i]) + e * H[i])) for i in idx))
        q1 = {i: _expit(_logit(Q1[i]) + e / g[i]) for i in idx}
        q0 = {i: _expit(_logit(Q0[i]) - e / (1 - g[i])) for i in idx}
        pf = statistics.fmean(q1[i] - q0[i] for i in idx)
        psis.append(pf)
        for i in idx:
            ic[i] = H[i] * (Y[i] - (q1[i] if A[i] else q0[i])) + q1[i] - q0[i] - pf
    return statistics.fmean(psis), math.sqrt(sum(t * t for t in ic)) / n, psis


def test_tmlcvc_basic():
    """Estimate, fold estimates and IC standard error match an independent
    bisection recomputation of the fold-wise targeting step."""
    Y, A, QA, Q1, Q0, g = _data()
    fold = [k % 3 for k in range(len(Y))]
    psi, se, psis = _cv(Y, A, Q1, Q0, g, fold)
    r = tmle_cv_targeting(Y, A, Q0, Q1, g, fold)
    assert r["estimate"] == pytest.approx(psi, abs=1e-9)
    assert r["se"] == pytest.approx(se, rel=1e-9)
    assert r["psi_fold"] == pytest.approx(psis, abs=1e-9)
    assert r["n_folds"] == 3


def test_tmlcvc_edge():
    """Out-of-range outcomes and propensities on the boundary are rejected;
    a single fold is the plain one-step-epsilon TMLE."""
    Y, A, QA, Q1, Q0, g = _data()
    with pytest.raises(ValueError):
        cvtmle([2.0] + Y[1:], A, Q0, Q1, g, [0] * len(Y))
    with pytest.raises(ValueError):
        cvtmle(Y, A, Q0, Q1, [1.0] + g[1:], [0] * len(Y))
    psi, se, _ = _cv(Y, A, Q1, Q0, g, [0] * len(Y))
    assert cvtmle(Y, A, Q0, Q1, g, [0] * len(Y))["estimate"] == pytest.approx(psi, abs=1e-9)
