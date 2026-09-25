"""Tests for tmlcom.tmle_compositional (TMLE on clr coordinates)."""

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

from morie.fn.tmlcom import tmle_compositional


def _comp_data(n=40):
    W = [math.sin(1.7 * k) for k in range(n)]
    g1 = [_expit(0.3 + 0.8 * w) for w in W]
    A = [1.0 if ((37 * k + 11) % 97 + 0.5) / 97.0 < g else 0.0 for k, g in enumerate(g1)]
    Y, Q1, Q0 = [], [], []
    for k, (a, w) in enumerate(zip(A, W)):
        u = ((53 * k + 7) % 89 + 0.5) / 89.0
        raw = [1.0 + 0.5 * a + 0.2 * w + 0.3 * u, 2.0 - 0.3 * a + 0.1 * u, 1.5 + 0.4 * w]
        Y.append(raw)

        def clr(v):
            L = [math.log(x) for x in v]
            m = sum(L) / len(L)
            return [x - m for x in L]
        Q1.append(clr([1.5 + 0.2 * w, 1.7, 1.5 + 0.4 * w]))
        Q0.append(clr([1.0 + 0.2 * w, 2.0, 1.5 + 0.4 * w]))
    return Y, A, Q1, Q0, [_expit(0.2 + 0.7 * w) for w in W]


def test_tmlcom_basic():
    """Each clr coordinate is scaled to [0, 1] over the observed values
    and predictions, targeted like a bounded outcome, rescaled; the
    effects are then centred to sum to zero (a clr contrast must)."""
    Y, A, Q1, Q0, g = _comp_data()
    n, D = len(Y), 3
    Z = []
    for r in Y:
        L = [math.log(v) for v in r]
        m = sum(L) / D
        Z.append([v - m for v in L])
    eff = []
    for j in range(D):
        col = [Z[i][j] for i in range(n)]
        pred = [Q1[i][j] for i in range(n)] + [Q0[i][j] for i in range(n)]
        a, b = min(col + pred), max(col + pred)
        ys = [(v - a) / (b - a) for v in col]
        q1 = [min(max((Q1[i][j] - a) / (b - a), 1e-6), 1 - 1e-6) for i in range(n)]
        q0 = [min(max((Q0[i][j] - a) / (b - a), 1e-6), 1 - 1e-6) for i in range(n)]
        qa = [q1[i] if A[i] else q0[i] for i in range(n)]
        mu1, mu0, *_ = _target(ys, A, qa, q1, q0, g)
        eff.append((mu1 - mu0) * (b - a))
    m = sum(eff) / D
    r = tmle_compositional(Y, A, Q1, Q0, g)
    assert list(r["effect"]) == pytest.approx([e - m for e in eff], abs=1e-8)
    assert abs(r["sum_effect"]) < 1e-12


def test_tmlcom_edge():
    """The perturbation is the closure of exp(effect); a non-positive
    part is refused."""
    Y, A, Q1, Q0, g = _comp_data()
    r = tmle_compositional(Y, A, Q1, Q0, g)
    ex = [math.exp(v) for v in r["effect"]]
    assert list(r["perturbation"]) == pytest.approx([v / sum(ex) for v in ex], rel=1e-14)
    Y[0][0] = 0.0
    with pytest.raises(ValueError):
        tmle_compositional(Y, A, Q1, Q0, g)
