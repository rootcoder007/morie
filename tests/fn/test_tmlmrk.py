"""Tests for tmlmrk.tmle_markov (average reward of a policy in an MDP)."""

import math

import pytest

from morie.fn.tmlmrk import tmle_markov


def _traj(n=400):
    """A three-state trajectory from a fixed LCG: the next state depends
    on the current state and action, the reward on both plus noise."""
    u = 12345
    s, S, A, R = 0, [], [], []
    for _ in range(n):
        u = (1103515245 * u + 12345) % 2 ** 31
        a = 1 if u / 2 ** 31 < 0.4 + 0.2 * s else 0
        u = (1103515245 * u + 12345) % 2 ** 31
        r = 0.5 * s + a - 0.3 * s * a + (u / 2 ** 31 - 0.5)
        S.append(s)
        A.append(a)
        R.append(r)
        u = (1103515245 * u + 12345) % 2 ** 31
        s = (s + 1 + a) % 3 if u / 2 ** 31 < 0.7 else s
    return S, A, R


def _reference(S, A, R, pol):
    """Plug-in: cell-mean reward and empirical transitions under pi;
    stationary law by power iteration; differential value
    h = sum_t P^t (r - V) (a convergent series for an aperiodic chain,
    and it automatically satisfies d'h = 0); IC per transition
    (d/d_b) 1{A=pi(S)}/b (R + h(S') - h(S) - V)."""
    n, ns = len(S), 3
    b = [sum(1 for j in range(n) if S[j] == s and A[j] == pol[s]) / sum(1 for j in range(n) if S[j] == s)
         for s in range(ns)]
    r = [sum(R[i] for i in range(n) if S[i] == s and A[i] == pol[s])
         / sum(1 for i in range(n) if S[i] == s and A[i] == pol[s]) for s in range(ns)]
    P = [[0.0] * ns for _ in range(ns)]
    for i in range(n - 1):
        if A[i] == pol[S[i]]:
            P[S[i]][S[i + 1]] += 1
    P = [[x / sum(row) for x in row] for row in P]
    d = [1.0 / ns] * ns
    for _ in range(5000):
        d = [sum(d[j] * P[j][k] for j in range(ns)) for k in range(ns)]
    V = sum(x * y for x, y in zip(d, r))
    h, term = [0.0] * ns, [x - V for x in r]
    for _ in range(5000):
        h = [x + y for x, y in zip(h, term)]
        term = [sum(P[k][j] * term[j] for j in range(ns)) for k in range(ns)]
    db = [sum(1 for i in range(n - 1) if S[i] == s) / (n - 1) for s in range(ns)]
    D = [(d[S[i]] / db[S[i]]) * (A[i] == pol[S[i]]) / b[S[i]] * (R[i] + h[S[i + 1]] - h[S[i]] - V)
         for i in range(n - 1)]
    return V, math.sqrt(sum(x * x for x in D)) / (n - 1)


def test_tmlmrk_basic():
    """Estimate and martingale standard error match the independent
    power-iteration / series recomputation.  The fluctuation is exactly
    zero because the cell-mean reward is saturated in (s, pi(s))."""
    S, A, R = _traj()
    pol = [1, 0, 1]
    V, se = _reference(S, A, R, pol)
    out = tmle_markov(S, A, R, pol)
    assert out["eps"] == pytest.approx(0.0, abs=1e-12)
    assert out["estimate"] == pytest.approx(V, abs=1e-9)
    assert out["se"] == pytest.approx(se, rel=1e-8)
    assert out["n_states"] == 3.0


def test_tmlmrk_edge():
    """A length-1 trajectory and a policy of the wrong length raise."""
    with pytest.raises(ValueError):
        tmle_markov([0], [1], [1.0], [1])
    S, A, R = _traj(50)
    with pytest.raises(ValueError):
        tmle_markov(S, A, R, [1, 0])
