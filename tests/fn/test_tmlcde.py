"""Tests for tmlcde.tmle_controlled_direct (van der Laan & Petersen 2008)."""

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


from morie.fn.tmlcde import tmle_controlled_direct


def _cde_data(n=40):
    W = [math.sin(1.7 * k) for k in range(n)]
    g1 = [_expit(0.3 + 0.8 * w) for w in W]
    A = [1.0 if ((37 * k + 11) % 97 + 0.5) / 97.0 < g else 0.0 for k, g in enumerate(g1)]
    h1 = [_expit(0.5 + 0.6 * w) for w in W]         # P(M = 1 | A = 1, W)
    h0 = [_expit(-0.4 + 0.6 * w) for w in W]        # P(M = 1 | A = 0, W)
    M = [1.0 if ((29 * k + 3) % 83 + 0.5) / 83.0 < (h1[k] if A[k] else h0[k]) else 0.0
         for k in range(n)]
    Y = [1.0 if ((53 * k + 7) % 89 + 0.5) / 89.0 < _expit(-0.5 + 0.8 * a + 0.6 * mm + 0.5 * w) else 0.0
         for k, (a, mm, w) in enumerate(zip(A, M, W))]
    Q1 = [_expit(-0.4 + 0.8 + 0.6 + 0.4 * w) for w in W]
    Q0 = [_expit(-0.4 + 0.6 + 0.4 * w) for w in W]
    QAM = [_expit(-0.4 + 0.8 * a + 0.6 * mm + 0.4 * w) for a, mm, w in zip(A, M, W)]
    return Y, A, M, QAM, Q1, Q0, g1, h1, h0


def test_tmlcde_basic():
    """H_a = 1{A=a, M=m} / (g_a h_m(a, W)); the two fluctuations have
    disjoint support, so each epsilon solves its own score equation
    (bisection here); Q*(a, m, W) moves along 1 / (g_a h_m(a, W)) at
    every unit; psi = mean Q*(1) - mean Q*(0)."""
    Y, A, M, QAM, Q1, Q0, g1, h1, h0 = _cde_data()
    n = len(Y)
    at = [1.0 if mm == 1.0 else 0.0 for mm in M]
    H1 = [t * a / (g * h) for t, a, g, h in zip(at, A, g1, h1)]
    H0 = [t * (1 - a) / ((1 - g) * h) for t, a, g, h in zip(at, A, g1, h0)]

    def solve(H):
        idx = [i for i in range(n) if H[i] > 0]

        def score(e):
            return sum(H[i] * (Y[i] - _expit(_logit(QAM[i]) + e * H[i])) for i in idx)
        lo, hi = -50.0, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if score(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)
    e1, e0 = solve(H1), solve(H0)
    Q1s = [_expit(_logit(q) + e1 / (g * h)) for q, g, h in zip(Q1, g1, h1)]
    Q0s = [_expit(_logit(q) + e0 / ((1 - g) * h)) for q, g, h in zip(Q0, g1, h0)]
    r = tmle_controlled_direct(Y, A, M, QAM, Q1, Q0, g1, [h1[i] if A[i] else h0[i] for i in range(n)],
                               m=1, hm1W=h1, hm0W=h0)
    assert r["estimate"] == pytest.approx(statistics.fmean(Q1s) - statistics.fmean(Q0s), abs=1e-9)
    assert r["max_weight"] == pytest.approx(max(max(H1), max(H0)), rel=1e-14)


def test_tmlcde_edge():
    """Mismatched lengths are refused."""
    Y, A, M, QAM, Q1, Q0, g1, h1, h0 = _cde_data()
    with pytest.raises(ValueError):
        tmle_controlled_direct(Y[:-1], A, M, QAM, Q1, Q0, g1, h1)
