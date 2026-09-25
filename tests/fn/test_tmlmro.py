"""Tests for tmlmro.tmle_marginal_or (TMLE of the marginal odds ratio, tmle 2.1.1)."""

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

from morie.fn.tmlmro import tmle_marginal_or


def test_tmlmro_basic():
    """psi = [mu1/(1-mu1)] / [mu0/(1-mu0)] and the delta-method log-OR
    influence curve IC1/(mu1(1-mu1)) - IC0/(mu0(1-mu0)), recomputed."""
    Y, A, QA, Q1, Q0, g = _data()
    mu1, mu0, IC1, IC0, *_ = _target(Y, A, QA, Q1, Q0, g)
    r = tmle_marginal_or(Y, A, QA, Q1, Q0, g)
    psi = (mu1 / (1 - mu1)) / (mu0 / (1 - mu0))
    ic = [a / (mu1 * (1 - mu1)) - b / (mu0 * (1 - mu0)) for a, b in zip(IC1, IC0)]
    assert r["estimate"] == pytest.approx(psi, rel=1e-8)
    assert r["log_or"] == pytest.approx(math.log(psi), abs=1e-8)
    assert r["se_log"] == pytest.approx(math.sqrt(statistics.variance(ic) / len(ic)), rel=1e-7)


def test_tmlmro_edge():
    """The CI is exp(log psi -+ z se_log): symmetric on the log scale."""
    Y, A, QA, Q1, Q0, g = _data()
    r = tmle_marginal_or(Y, A, QA, Q1, Q0, g)
    assert math.log(r["ci_upper"]) - r["log_or"] == pytest.approx(r["log_or"] - math.log(r["ci_lower"]), rel=1e-12)
