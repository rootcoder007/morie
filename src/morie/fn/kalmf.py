# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Kalman filter for the linear Gaussian state space model.

Kalman (1960), "A new approach to linear filtering and prediction
problems", Trans. ASME J. Basic Engineering 82(1):35-45,
doi:10.1115/1.3662552.  With

    x_t = F x_{t-1} + w_t,   w ~ N(0, Q),
    y_t = H x_t     + v_t,   v ~ N(0, R),

the recursion alternates a prediction and an update:

    x_{t|t-1} = F x_{t-1|t-1},        P_{t|t-1} = F P F' + Q,
    S_t = H P_{t|t-1} H' + R,         K_t = P_{t|t-1} H' S_t^{-1},
    x_{t|t} = x_{t|t-1} + K_t v_t,    P_{t|t} = (I - K_t H) P_{t|t-1},

with innovation v_t = y_t - H x_{t|t-1}.  The prediction error
decomposition gives the exact log-likelihood as a by-product.  With
Q = 0, F = I, H = I and P_0 = R the filtered state is exactly the
running mean of the prior mean and the observations so far, which is
the closed form the tests check.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kalman_filter"]


def _obs(y):
    Y = core.mat(y)
    return Y, len(Y), len(Y[0])


def _solve_cols(S, B):
    """Return S^{-1} B, column by column, for symmetric positive definite S."""
    m = len(S)
    p = len(B[0])
    out = [[0.0] * p for _ in range(m)]
    for j in range(p):
        col = core.cholsolve(S, [B[i][j] for i in range(m)])
        for i in range(m):
            out[i][j] = col[i]
    return out


def _filter(Y, F, H, Q, R, x0, P0):
    n = len(Y)
    d = len(F)
    m = len(H)
    x = list(x0)
    P = [row[:] for row in P0]
    xs = []
    Ps = []
    xp = []
    Pp = []
    ll = 0.0
    for t in range(n):
        xpred = core.matvec(F, x)
        Ppred = core.matmul(core.matmul(F, P), [[F[j][i] for j in range(d)] for i in range(d)])
        Ppred = [[Ppred[i][j] + Q[i][j] for j in range(d)] for i in range(d)]
        HP = core.matmul(H, Ppred)
        S = [[sum(HP[i][k] * H[j][k] for k in range(d)) + R[i][j] for j in range(m)] for i in range(m)]
        v = [Y[t][i] - sum(H[i][k] * xpred[k] for k in range(d)) for i in range(m)]
        SinvHP = _solve_cols(S, HP)
        K = [[SinvHP[i][j] for i in range(m)] for j in range(d)]
        x = [xpred[j] + sum(K[j][i] * v[i] for i in range(m)) for j in range(d)]
        KH = core.matmul(K, H)
        P = [[Ppred[i][j] - sum(KH[i][k] * Ppred[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
        Sinv_v = core.cholsolve(S, v)
        quad = sum(v[i] * Sinv_v[i] for i in range(m))
        L = core.chol(S)
        logdet = 2.0 * sum(math.log(L[i][i]) for i in range(m))
        ll += -0.5 * (m * math.log(2.0 * math.pi) + logdet + quad)
        xs.append(list(x))
        Ps.append([row[:] for row in P])
        xp.append(list(xpred))
        Pp.append([row[:] for row in Ppred])
    return xs, Ps, xp, Pp, ll


def kalman_filter(y, F, H, Q, R, x0=None, P0=None):
    """Filtered states, covariances and exact log-likelihood."""
    Y, n, m = _obs(y)
    if n == 0:
        raise ValueError("kalman_filter: y is empty")
    Fm = core.mat(F)
    Hm = core.mat(H)
    Qm = core.mat(Q)
    Rm = core.mat(R)
    d = len(Fm)
    if len(Fm[0]) != d:
        raise ValueError("kalman_filter: F must be square")
    if len(Hm) != m or len(Hm[0]) != d:
        raise ValueError("kalman_filter: H must be m x d")
    if len(Qm) != d or len(Rm) != m:
        raise ValueError("kalman_filter: Q must be d x d and R m x m")
    xi = [0.0] * d if x0 is None else core.vec(x0)
    Pi = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)] if P0 is None else core.mat(P0)
    if len(xi) != d or len(Pi) != d:
        raise ValueError("kalman_filter: x0 and P0 must match the state dimension")
    xs, Ps, xp, Pp, ll = _filter(Y, Fm, Hm, Qm, Rm, xi, Pi)
    return RichResult(
        title="Kalman filter",
        summary_lines=[("n", n), ("state dim", d), ("loglik", ll)],
        payload={
            "estimate": xs[-1][0],
            "state": xs,
            "cov": Ps,
            "predicted": xp,
            "predicted_cov": Pp,
            "loglik": ll,
            "n": n,
            "method": "predict/update recursion of Kalman (1960) with the prediction error decomposition",
        },
    )


def cheatsheet():
    return "kalmf: Kalman filter"
