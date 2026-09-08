# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Rauch-Tung-Striebel fixed-interval smoother.

Rauch, Tung and Striebel (1965), "Maximum likelihood estimates of
linear dynamic systems", AIAA Journal 3(8):1445-1450,
doi:10.2514/3.3166.  A forward Kalman pass is followed by the backward
recursion

    C_t = P_{t|t} F' P_{t+1|t}^{-1},
    x_{t|n} = x_{t|t} + C_t (x_{t+1|n} - x_{t+1|t}),
    P_{t|n} = P_{t|t} + C_t (P_{t+1|n} - P_{t+1|t}) C_t'.

With Q = 0 and F = I the state is constant, so every smoothed state
equals the final filtered state -- an exact identity the tests use.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kalman_smoother"]


def _fwd(Y, F, H, Q, R, x, P):
    n = len(Y)
    d = len(F)
    m = len(H)
    xs = []
    Ps = []
    xp = []
    Pp = []
    ll = 0.0
    for t in range(n):
        xpred = core.matvec(F, x)
        FP = core.matmul(F, P)
        Ppred = [[sum(FP[i][k] * F[j][k] for k in range(d)) + Q[i][j] for j in range(d)] for i in range(d)]
        HP = core.matmul(H, Ppred)
        S = [[sum(HP[i][k] * H[j][k] for k in range(d)) + R[i][j] for j in range(m)] for i in range(m)]
        v = [Y[t][i] - sum(H[i][k] * xpred[k] for k in range(d)) for i in range(m)]
        K = [[0.0] * m for _ in range(d)]
        for j in range(d):
            col = core.cholsolve(S, [HP[i][j] for i in range(m)])
            for i in range(m):
                K[j][i] = col[i]
        x = [xpred[j] + sum(K[j][i] * v[i] for i in range(m)) for j in range(d)]
        KH = core.matmul(K, H)
        P = [[Ppred[i][j] - sum(KH[i][k] * Ppred[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
        sv = core.cholsolve(S, v)
        L = core.chol(S)
        ll += -0.5 * (m * math.log(2.0 * math.pi) + 2.0 * sum(math.log(L[i][i]) for i in range(m)) + sum(v[i] * sv[i] for i in range(m)))
        xs.append(list(x))
        Ps.append([r[:] for r in P])
        xp.append(list(xpred))
        Pp.append([r[:] for r in Ppred])
    return xs, Ps, xp, Pp, ll


def _smooth(xs, Ps, xp, Pp, F, ridge):
    n = len(xs)
    d = len(F)
    xsm = [list(v) for v in xs]
    Psm = [[r[:] for r in M] for M in Ps]
    for t in range(n - 2, -1, -1):
        A = [[Pp[t + 1][i][j] + (ridge if i == j else 0.0) for j in range(d)] for i in range(d)]
        PF = [[sum(Ps[t][i][k] * F[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
        C = [[0.0] * d for _ in range(d)]
        for i in range(d):
            C[i] = core.cholsolve(A, PF[i])
        dx = [xsm[t + 1][j] - xp[t + 1][j] for j in range(d)]
        xsm[t] = [xs[t][j] + sum(C[j][k] * dx[k] for k in range(d)) for j in range(d)]
        dP = [[Psm[t + 1][i][j] - Pp[t + 1][i][j] for j in range(d)] for i in range(d)]
        CdP = core.matmul(C, dP)
        Psm[t] = [[Ps[t][i][j] + sum(CdP[i][k] * C[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
    return xsm, Psm


def kalman_smoother(y, F, H, Q, R, x0=None, P0=None, ridge=1e-12):
    """Smoothed states from the forward filter and the RTS backward pass."""
    Y = core.mat(y)
    n = len(Y)
    if n == 0:
        raise ValueError("kalman_smoother: y is empty")
    m = len(Y[0])
    Fm = core.mat(F)
    Hm = core.mat(H)
    Qm = core.mat(Q)
    Rm = core.mat(R)
    d = len(Fm)
    if len(Hm) != m or len(Hm[0]) != d:
        raise ValueError("kalman_smoother: H must be m x d")
    x = [0.0] * d if x0 is None else core.vec(x0)
    P = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)] if P0 is None else core.mat(P0)
    xs, Ps, xp, Pp, ll = _fwd(Y, Fm, Hm, Qm, Rm, x, P)
    xsm, Psm = _smooth(xs, Ps, xp, Pp, Fm, float(ridge))
    return RichResult(
        title="Rauch-Tung-Striebel smoother",
        summary_lines=[("n", n), ("state dim", d)],
        payload={
            "estimate": xsm[0][0],
            "smoothed": xsm,
            "smoothed_cov": Psm,
            "filtered": xs,
            "loglik": ll,
            "n": n,
            "method": "forward Kalman pass plus the RTS backward recursion, Rauch, Tung & Striebel (1965)",
        },
    )


def cheatsheet():
    return "kalmS: Rauch-Tung-Striebel smoother"


# compact alias per ledger/NAMING.md
kalmansmoother = kalman_smoother
