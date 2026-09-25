"""Tests for tmlrct.tmle_rct_assisted (trial ATE, observational rows in Q only)."""

import math
import statistics

import pytest

from morie.fn.tmlrct import tmle_rct_assisted


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _data(n1=40, n2=60):
    n = n1 + n2
    X = [[math.sin(1.3 * k), math.cos(0.7 * k)] for k in range(n)]
    D = [float(k % 2) if k < n1 else (1.0 if X[k][0] > 0 else 0.0) for k in range(n)]
    y = [0.5 + 1.0 * d + 0.4 * x[0] - 0.2 * x[1] + (0.3 if k >= n1 else 0.0) + 0.25 * math.sin(9.1 * k)
         for k, (d, x) in enumerate(zip(D, X))]
    return y[:n1], y[n1:], D, X


def _reference(y1, y2, D, X):
    """Q by OLS of Y on (D, 1, X, S) over all rows; g the trial's own
    treated fraction; H = S/P(S) (D/g - (1-D)/(1-g)); linear fluctuation
    eps = sum H (Y - Q) / sum H^2; psi the trial mean of Q1* - Q0*;
    IC = H (Y - Q*) + S/P(S) (Q1* - Q0* - psi)."""
    y = y1 + y2
    n1, n = len(y1), len(y)
    S = [1.0 if i < n1 else 0.0 for i in range(n)]
    Z = [[D[i], 1.0] + X[i] + [S[i]] for i in range(n)]
    q = len(Z[0])
    b = _solve([[sum(z[r] * z[c] for z in Z) for c in range(q)] for r in range(q)],
               [sum(z[r] * t for z, t in zip(Z, y)) for r in range(q)])
    Q1 = [sum(u * v for u, v in zip([1.0] + z[1:], b)) for z in Z]
    Q0 = [sum(u * v for u, v in zip([0.0] + z[1:], b)) for z in Z]
    g = statistics.fmean(D[:n1])
    pt = n1 / n
    H = [s / pt * (d / g - (1 - d) / (1 - g)) for s, d in zip(S, D)]
    QA = [a if d else c for d, a, c in zip(D, Q1, Q0)]
    e = sum(h * (t - qa) for h, t, qa in zip(H, y, QA)) / sum(h * h for h in H)
    Q1s = [a + e * s / (pt * g) for a, s in zip(Q1, S)]
    Q0s = [c - e * s / (pt * (1 - g)) for c, s in zip(Q0, S)]
    psi = statistics.fmean(Q1s[i] - Q0s[i] for i in range(n1))
    ic = [H[i] * (y[i] - QA[i] - e * H[i]) + S[i] / pt * (Q1s[i] - Q0s[i] - psi) for i in range(n)]
    return psi, e, g, math.sqrt(statistics.variance(ic) / n)


def test_tmlrct_basic():
    """Estimate, fluctuation, trial propensity and IC standard error
    match the independent recomputation."""
    y1, y2, D, X = _data()
    psi, e, g, se = _reference(y1, y2, D, X)
    r = tmle_rct_assisted(y1, y2, D, X)
    assert r["estimate"] == pytest.approx(psi, abs=1e-9)
    assert r["eps"] == pytest.approx(e, abs=1e-9)
    assert r["g_rct"] == pytest.approx(g, abs=1e-15)
    assert r["se"] == pytest.approx(se, rel=1e-9)
    assert (r["n_rct"], r["n_obs"]) == (40.0, 60.0)


def test_tmlrct_edge():
    """One trial row and wrongly stacked D or X raise."""
    y1, y2, D, X = _data()
    with pytest.raises(ValueError):
        tmle_rct_assisted(y1[:1], y2, D[39:], X[39:])
    with pytest.raises(ValueError):
        tmle_rct_assisted(y1, y2, D[:-1], X)
    with pytest.raises(ValueError):
        tmle_rct_assisted(y1, y2, D, X[:-1])
