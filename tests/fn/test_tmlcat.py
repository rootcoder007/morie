"""Tests for tmlcat.tmlecat (TMLE for a categorical treatment)."""

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

from morie.fn.tmlcat import tmlecat, tmle_categorical_outcome


def _cat(n=45):
    W = [math.sin(1.3 * k) for k in range(n)]
    A = [1 + (7 * k + int(3 * (w + 1))) % 3 for k, w in enumerate(W)]
    Y = [1.0 if ((29 * k + 5) % 83 + 0.5) / 83.0 < _expit(-0.4 + 0.3 * a + 0.8 * w) else 0.0
         for k, (a, w) in enumerate(zip(A, W))]
    Q = [[_expit(-0.2 + 0.25 * a + 0.5 * w) for a in (1, 2, 3)] for w in W]
    G = []
    for w in W:
        u = [math.exp(0.2 * w), math.exp(-0.3 * w), 1.0]
        G.append([v / sum(u) for v in u])
    return Y, A, Q, G


def _level(Y, A, Q, G, a):
    """Logistic fluctuation of logit Q(A,W) on H_a = 1{A=a}/g_a; only the
    A = a rows carry H_a, so the score is monotone in epsilon."""
    n = len(Y)
    idx = [i for i in range(n) if A[i] == a]
    g = [G[i][a - 1] for i in range(n)]

    def score(e):
        return sum((Y[i] - _expit(_logit(Q[i][a - 1]) + e / g[i])) / g[i] for i in idx)
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if score(mid) > 0 else (lo, mid)
    e = 0.5 * (lo + hi)
    Qa = [_expit(_logit(Q[i][a - 1]) + e / g[i]) for i in range(n)]
    psi = statistics.fmean(Qa)
    ic = [((Y[i] - Qa[i]) / g[i] if A[i] == a else 0.0) + Qa[i] - psi for i in range(n)]
    return psi, ic


def test_tmlcat_basic():
    """psi_a, its IC standard error and the contrast standard errors
    match an independent per-level bisection."""
    Y, A, Q, G = _cat()
    n = len(Y)
    fits = [_level(Y, A, Q, G, a) for a in (1, 2, 3)]
    r = tmle_categorical_outcome(Y, A, Q, G, ref=1, gbound=0.0)
    for a in range(3):
        psi, ic = fits[a]
        assert r["psi"][a] == pytest.approx(psi, abs=1e-9)
        assert r["se"][a] == pytest.approx(math.sqrt(statistics.variance(ic) / n), rel=1e-8)
        d = [x - y for x, y in zip(ic, fits[0][1])]
        assert r["contrast"][a] == pytest.approx(psi - fits[0][0], abs=1e-9)
        if a:
            assert r["contrast_se"][a] == pytest.approx(math.sqrt(statistics.variance(d) / n), rel=1e-8)
    assert r["contrast"][0] == 0.0 and r["L"] == 3.0


def test_tmlcat_edge():
    """Zero-based labels, a single level and outcomes outside [0,1] raise."""
    Y, A, Q, G = _cat()
    with pytest.raises(ValueError):
        tmlecat(Y, [a - 1 for a in A], Q, G)
    with pytest.raises(ValueError):
        tmlecat(Y, [1] * len(Y), [[q[0]] for q in Q], [[1.0]] * len(Y))
    with pytest.raises(ValueError):
        tmlecat([2.0] + Y[1:], A, Q, G)
